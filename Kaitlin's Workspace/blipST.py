from pathlib import Path
from typing import List, Optional
import torch
import os
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time

start_time = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages.

cwd = Path.cwd()
parent_directory = cwd.parent

device = "cuda" if torch.cuda.is_available() else "cpu"

ARASAAC_DATA_PATH = f'{parent_directory}/data/arasaac_pictograms_complete_20251106_130529.json'
SYNSET_MAP_PATH = f'{parent_directory}/data/arasaac_synset_mapping_20251106_130530.csv'



# Small & fast. Swap to "Salesforce/blip-image-captioning-large" for a bit more quality.
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(device)

def caption_image(image: Image.Image, prompt: Optional[str] = None, max_length: int = 30, num_beams: int = 5, repetition_penalty: float = 1.15):
    """
    If `prompt` is given, BLIP treats it like a guiding prefix (e.g., "A detailed photo of").
    """
    image = image.convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
    return processor.decode(out[0], skip_special_tokens=True).strip()

img_path = f'{cwd}/test_images/cleaned_test_images/airport/clean_airport_002.jpg'
image = Image.open(img_path)

caption = caption_image(image, prompt="A concise description of the scene:")

print("BLIP caption:")
print(caption_image(image))
# display(image)
image.show() # from PyCharm

print(caption)

df = pd.read_csv(SYNSET_MAP_PATH)
df = df.drop_duplicates(subset=['primary_keyword'])

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = caption
sentence_emb = model.encode(sentence, convert_to_tensor=True)

df["keyword_emb"] = df["primary_keyword"].apply(
    lambda x: model.encode(x, convert_to_tensor=True))

df["score"] = df["keyword_emb"].apply(
    lambda emb: util.cos_sim(sentence_emb, emb).item())

df_sorted = df.sort_values("score", ascending=False)
print(df_sorted[["primary_keyword", "score"]])

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Script took {elapsed_time:.2f} seconds")
print(f"Script took {elapsed_time/60:.2f} minutes")