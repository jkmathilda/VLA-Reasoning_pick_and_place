from transformers import AutoProcessor, AutoModelForImageTextToText
import os

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "sample_data/image.png"},
            {"type": "text", "text": "There are three bowls on the table. The bowls are red, green, and blue. Where is the blue bowl? Answer with left, middle, or right."},
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))