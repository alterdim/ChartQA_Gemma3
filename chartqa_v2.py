import torch
from transformers import pipeline
from PIL import Image
import json
from table_eval import is_answer_correct  # <-- function from the other script
from tqdm import tqdm


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
        image = Image.open(image_path).copy()
        shot0_img = Image.open(path_image("41699051005347.png")).copy()
        shot1_img = Image.open(path_image("41699051005347.png")).copy()
        shot2_img = Image.open(path_image("41810321001157.png")).copy()
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
                    {"type": "image", "image": shot0_img},
                    {"type": "text", "text": shot0_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": shot0_answer}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": shot1_img},
                    {"type": "text", "text": shot1_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": shot1_answer}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": shot2_img},
                    {"type": "text", "text": shot2_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": shot2_answer}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            },
        ]
        output = self.pipeline(template, max_new_tokens=150)
        return output[0]["generated_text"][-1]["content"]


# === MAIN EVAL LOOP ===
gemma = Gemma3()
correct = 0
total = 0

with open("ChartQA Dataset/test/test_human.json") as file:
    data = json.load(file)

for entry in tqdm(data):
    img_path = path_image(entry["imgname"])
    query = entry["query"]
    label = entry["label"]

    try:
        prediction = gemma.infer(query, img_path)
        is_correct = is_answer_correct(prediction, label)
        correct += int(is_correct)
        total += 1
        print(f"✓ {is_correct} | Q: {query} | GT: {label} | Pred: {prediction}")
    except Exception as e:
        print(f"⚠️ Error on {entry['imgname']}: {e}")
        total += 1  # count it as wrong if it fails

# Final score
accuracy = correct / total if total > 0 else 0
print(f"\n✅ Final Accuracy: {accuracy:.4f} ({correct}/{total})")
