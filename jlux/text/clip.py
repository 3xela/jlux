from transformers import CLIPTokenizer, CLIPTextModel
import torch
import jax.numpy as jnp

class CLIPWrapper:
    def __init__(self, path):
        self._path = path
        self.device = "cuda"
        self._tokenizer = CLIPTokenizer.from_pretrained(self._path)
        self._text_model = CLIPTextModel.from_pretrained(self._path).to(self.device).eval()

    def _tokenize(self, text, max_length =77, padding ="max_length", truncation=True, return_tensors="pt"):
        toks = self._tokenizer(text, max_length=max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)
        return toks
    
    def __call__(self, prompts: list[str]):
        toks = self._tokenize(prompts).to(self.device)
        with torch.no_grad():
            out = self._text_model(**toks).pooler_output
            return jnp.asarray(out.cpu().numpy())
        