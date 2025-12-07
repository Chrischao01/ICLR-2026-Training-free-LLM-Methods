from typing import Callable, Optional, Union, Dict, Any, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2PreTrainedModel,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs
from transformers.integrations.sdpa_attention import sdpa_attention_forward

logger = logging.get_logger(__name__)


class Qwen2InjectionAttention(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            global_prompt_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        is_injection_step = (
                injection_layer_idx is not None
                and self.layer_idx > 0
                and self.layer_idx <= 25
                and self.layer_idx % injection_layer_idx == 0
                and global_prompt_embedding is not None
                and q_len > 1
        )

        if is_injection_step:
            print(f"Injecting global prompt embedding at layer {self.layer_idx}...")
            query_states = self.q_proj(hidden_states)
            key_states_text = self.k_proj(hidden_states)
            value_states_text = self.v_proj(hidden_states)

            if global_prompt_embedding.dim() == 2:
                global_prompt_embedding = global_prompt_embedding.unsqueeze(1)  # [bsz, h_dim] -> [bsz, 1, h_dim]

            key_states_prompt = self.k_proj(global_prompt_embedding)
            value_states_prompt = self.v_proj(global_prompt_embedding)

            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            key_states_text = key_states_text.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            value_states_text = value_states_text.view(bsz, q_len, -1, self.head_dim).transpose(1,
                                                                                                                      2)

            prompt_len = global_prompt_embedding.shape[1]
            key_states_prompt = key_states_prompt.view(bsz, prompt_len, -1,
                                                       self.head_dim).transpose(1, 2)
            value_states_prompt = value_states_prompt.view(bsz, prompt_len, -1,
                                                           self.head_dim).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states_text = apply_rotary_pos_emb(query_states, key_states_text, cos, sin)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states_text, value_states_text = past_key_value.update(key_states_text, value_states_text,
                                                                           self.layer_idx, cache_kwargs)

            query_first = query_states[:, :, :1, :]

            key_prompt_repeated = repeat_kv(key_states_prompt, self.num_key_value_groups)
            value_prompt_repeated = repeat_kv(value_states_prompt, self.num_key_value_groups)

            attn_output_first = F.scaled_dot_product_attention(
                query_first,
                key_prompt_repeated,
                value_prompt_repeated,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

            query_others = query_states[:, :, 1:, :]
            key_text_repeated = repeat_kv(key_states_text, self.num_key_value_groups)
            value_text_repeated = repeat_kv(value_states_text, self.num_key_value_groups)

            causal_mask_others = attention_mask[:, :, 1:, :] if attention_mask is not None else None

            attn_output_others = F.scaled_dot_product_attention(
                query_others,
                key_text_repeated,
                value_text_repeated,
                attn_mask=causal_mask_others,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=(attention_mask is None and q_len > 1),
            )

            attn_output = torch.cat([attn_output_first, attn_output_others], dim=2)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            return attn_output, None

        else:
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )



class Qwen2InjectionDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2InjectionAttention(config=config, layer_idx=layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            global_prompt_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            global_prompt_embedding=global_prompt_embedding,
            injection_layer_idx=injection_layer_idx,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states



class Qwen2ModelWithPromptInjection(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2InjectionDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        global_prompt_embedding = None
        if (
                injection_layer_idx is not None and
                inputs_embeds.shape[1] > 1
        ):
            global_prompt_embedding = inputs_embeds


        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                global_prompt_embedding=global_prompt_embedding,
                injection_layer_idx=injection_layer_idx,
                **kwargs,
            )

            if (
                    injection_layer_idx is not None and
                    hidden_states.shape[1] > 1 and
                    (idx + 1) % injection_layer_idx == 0 and
                    idx != (self.config.num_hidden_layers - 1) 
            ):
                logger.info(f"Updating global prompt embedding after layer {idx}")
                global_prompt_embedding = hidden_states

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )



class Qwen2ForCausalLMWithPromptInjection(Qwen2ForCausalLM):
    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Qwen2ModelWithPromptInjection(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, Any]:
        injection_layer_idx = kwargs.pop("injection_layer_idx", None)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            **kwargs
        )

        model_inputs["injection_layer_idx"] = injection_layer_idx
        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            injection_layer_idx: Optional[int] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:



        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            injection_layer_idx=injection_layer_idx,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )