import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Any

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLCausalLMOutputWithPast,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.cache_utils import Cache,DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput

logger = logging.get_logger(__name__)


class Qwen2_5_VLInjectionAttention(Qwen2_5_VLAttention):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            global_image_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, q_len, _ = hidden_states.size()

        is_injection_step = (
                injection_layer_idx is not None
                and self.layer_idx % injection_layer_idx == 0
                and self.layer_idx <= 25
                and self.layer_idx != 0
                and global_image_embedding is not None
                and q_len > 1
        )

        if is_injection_step:
            print(f"injecting in {self.layer_idx} layer...")
            if output_attentions:
                logger.warning_once(
                    "`output_attentions=True` is not supported in injection mode. Returning None for attentions.")

            query_states = self.q_proj(hidden_states)
            key_states_text = self.k_proj(hidden_states)
            value_states_text = self.v_proj(hidden_states)

            if global_image_embedding.dim() == 2:
                global_image_embedding = global_image_embedding.unsqueeze(1)
            key_states_image = self.k_proj(global_image_embedding)
            value_states_image = self.v_proj(global_image_embedding)

            img_bsz, img_len, img_dim = global_image_embedding.shape

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states_text = key_states_text.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states_text = value_states_text.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,
                                                                                                                      2)

            key_states_image = key_states_image.view(bsz, img_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states_image = value_states_image.view(bsz, img_len, self.num_key_value_heads, self.head_dim).transpose(1,
                                                                                                                    2)

            cos, sin = position_embeddings
            query_states, key_states_text = apply_multimodal_rotary_pos_emb(
                query_states, key_states_text, cos, sin, self.rope_scaling["mrope_section"]
            )

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states_text, value_states_text = past_key_value.update(key_states_text, value_states_text,
                                                                           self.layer_idx, cache_kwargs)

            query_first = query_states[:, :, :1, :]

            key_image_repeated = repeat_kv(key_states_image, self.num_key_value_groups)
            value_image_repeated = repeat_kv(value_states_image, self.num_key_value_groups)

            attn_output_first = torch.nn.functional.scaled_dot_product_attention(
                query_first,
                key_image_repeated,
                value_image_repeated,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

            query_others = query_states[:, :, 1:, :]
            key_others = key_states_text
            value_others = value_states_text

            key_others_repeated = repeat_kv(key_others, self.num_key_value_groups)
            value_others_repeated = repeat_kv(value_others, self.num_key_value_groups)

            causal_mask_others = attention_mask[:, :, 1:, :] if attention_mask is not None else None
            is_causal = causal_mask_others is None and query_others.shape[2] > 1

            attn_output_others = torch.nn.functional.scaled_dot_product_attention(
                query_others,
                key_others_repeated,
                value_others_repeated,
                attn_mask=causal_mask_others,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal
            )

            attn_output = torch.cat([attn_output_first, attn_output_others], dim=2)
            attn_weights = None

        else:
            return super().forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                **kwargs,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Qwen2_5_VLInjectionDecoderLayer(Qwen2_5_VLDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2_5_VLInjectionAttention(config, layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            global_image_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            global_image_embedding=global_image_embedding,
            injection_layer_idx=injection_layer_idx,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen2_5_VLTextModelWithInjection(Qwen2_5_VLTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLInjectionDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            global_image_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            st_ed_idx = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, Any]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)


        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                global_image_embedding=global_image_embedding,
                injection_layer_idx=injection_layer_idx,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if global_image_embedding is not None and idx % injection_layer_idx == 0 and idx != 0:
                global_image_embedding = hidden_states[:, st_ed_idx[0]:st_ed_idx[1] + 1, :]


            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class Qwen2_5_VLModelWithInjection(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen2_5_VLTextModelWithInjection(config.text_config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        global_image_embedding_batch = None
        start_indices = None
        end_indices = None
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0)

            if input_ids is None:
                image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                image_mask = image_mask.all(-1)
            else:
                image_mask = input_ids == self.config.image_token_id

            n_image_tokens = (image_mask).sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            n_image_features = image_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


            if injection_layer_idx is not None:
                print(f"Preparing global image embedding for injection.")
                image_mask = (input_ids == self.config.image_token_id)
                start_indices = torch.argmax(image_mask.int(), dim=1)
                flipped_mask = torch.flip(image_mask, dims=[1])
                end_indices_rev = torch.argmax(flipped_mask.int(), dim=1)
                sequence_length = input_ids.shape[1]
                end_indices = sequence_length - 1 - end_indices_rev
                print("Start Indices:", start_indices)
                print("End Indices:", end_indices)
                global_image_embeddings_list = []
                single_image_embeds = image_embeds
                global_image_embeddings_list.append(single_image_embeds)

                if global_image_embeddings_list:
                    global_image_embedding_batch = torch.stack(global_image_embeddings_list, dim=0)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0)

            if input_ids is None:
                video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                video_mask = video_mask.all(-1)
            else:
                video_mask = input_ids == self.config.video_token_id

            n_video_tokens = (video_mask).sum()
            n_video_features = video_embeds.shape[0]
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                    (input_ids is not None and input_ids.shape[1] != 1)
                    or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                    (cache_position is not None and cache_position[0] == 0)
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            global_image_embedding=global_image_embedding_batch,
            injection_layer_idx=injection_layer_idx,
            st_ed_idx=(start_indices, end_indices),
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class Qwen2_5_VLForConditionalGenerationWithInjection(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLModelWithInjection(config)  

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            injection_layer_idx: Optional[int] = None,
            **kwargs: Unpack[Any],
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            injection_layer_idx=injection_layer_idx,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        injection_layer_idx = kwargs.pop("injection_layer_idx", None)
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["injection_layer_idx"] = injection_layer_idx

        return model_inputs