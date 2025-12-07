from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from sinktrack_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithInjection

INJECTION_LAYER = 5

model_path = '/home/resource/model/Qwen2.5-VL-7B-Instruct'
model = Qwen2_5_VLForConditionalGenerationWithInjection.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

def get_res(prompt, image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image[0],
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=True, injection_layer_idx=INJECTION_LAYER)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def main():
    prompt = "Describe this image."
    image = ['0.png']
    output = get_res(prompt, image)
    print(output)




if __name__ == '__main__':
    main()