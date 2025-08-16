# embedding_module.py

import re
import torch
from transformers import CLIPTokenizer, CLIPTextModel

class TextEmbedder:
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = None,
                 max_length: int = 77):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Use CLIP-specific classes for tokenizer & text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.,!?\-']", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        cleaned = [self.clean_text(t) for t in texts]
        batch = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                return_dict=True
            )
        return outputs.last_hidden_state, batch.attention_mask

if __name__ == "__main__":
    embedder = TextEmbedder()
    texts = ["A red apple on a table.", "A circle and a square."]
    embeddings, masks = embedder.encode(texts)
    print("Embeddings:", embeddings.shape)
    print("Mask:", masks.shape)
