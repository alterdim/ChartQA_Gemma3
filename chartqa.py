import torch
from transformers import pipeline
from PIL import Image
import json


def path_image(img):
    return "ChartQA Dataset/test/png/" + img


class Gemma3:
    def __init__(self):
        self.pipeline = pipeline(
            task="image-text-to-text",
            model="google/gemma-3-4b-it",
            device=0,
            torch_dtype=torch.bfloat16,
        )

    def infer(self, text, image_path):
        image = Image.open(image_path)
        shot0_path = Image.open(path_image("41699051005347.png"))
        shot1_path = Image.open(path_image("41699051005347.png"))
        shot2_path = Image.open(path_image("41810321001157.png"))
        shot0_prompt = "How many food item is shown in the bar graph?"
        shot1_prompt = "What is the difference in value between Lamb and Corn?"
        shot2_prompt = "How many bars are shown in the chart?"
        shot0_answer = "14"
        shot1_answer = "0.57"
        shot2_answer = "3"
        template = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "I will ask you questions about graphs. Reply with the answer only, no explanation. There is always an answer, so just reply.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "path": shot0_path,
                    },
                    {"type": "text", "text": shot0_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": shot0_answer},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "path": shot1_path,
                    },
                    {"type": "text", "text": shot1_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": shot1_answer},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "path": shot2_path,
                    },
                    {"type": "text", "text": shot2_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": shot2_answer},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "path": image,
                    },
                    {"type": "text", "text": text},
                ],
            },
        ]
        output = self.pipeline(template, max_new_tokens=150)
        shot0_path.close()
        shot1_path.close()
        shot2_path.close()
        image.close()
        return output[0]["generated_text"][-1]["content"]


gemma = Gemma3()
with open("ChartQA Dataset/test/test_human.json") as file:
    data = json.load(file)

for entry in data:
    imgth = "ChartQA Dataset/test/png/" + entry["imgname"]
    query = entry["query"]
    label = entry["label"]
    print(f"Image: {imgth}")
    print(f"Question: {query}")
    print(f"Answer: {label}")
    print("---")
    out = gemma.infer(query, imgth)
    print(out)
