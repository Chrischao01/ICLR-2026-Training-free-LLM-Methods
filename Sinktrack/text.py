import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sinktrack_qwen2_5 import Qwen2ForCausalLMWithPromptInjection

def load_model_and_tokenizer(model_id: str, device: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2ForCausalLMWithPromptInjection.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


model_path = "/home/resource/model/Qwen2.5-7B-Instruct"
injection_layer_idx = 5

model, tokenizer = load_model_and_tokenizer(model_path, device="cuda")


def get_response(model, tokenizer, prompt_text):
    messages = [
        {"role": "system",
         "content": "You are an expert at telling stories."},
        {"role": "user", "content": prompt_text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048,
        injection_layer_idx=injection_layer_idx
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


if __name__ == "__main__":
    prompt = "Tell me a good story!"
    output = get_response(model, tokenizer, prompt)
    print(output)