from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

class SceneCaptioner:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)

    def caption_image(self, image_path, prompt=None):
        """Generate detailed caption for AAC board generation."""
        image = Image.open(image_path).convert('RGB')

        # Prompt(s) to refine output.
        ## Try structured output.
        ## Try natural language interrogation.
        prompts = [
            "Question: Where is this scene taking place? Answer:"
            "Question: What is happening in this image? Answer:",
            "Question: Describe where this scene is taking place and all the people, objects, and activities in this scene. Answer:",
        ]

        captions = []
        for p in prompts:
            inputs = self.processor(image, text=p, return_tensors="pt").to(
                self.device, torch.float16 if self.device == 'cuda' else torch.float32
            )

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
                temperature=0.7
            )

            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            captions.append(caption)

        # Return longest caption (usually most detailed).
        return max(captions, key=len)