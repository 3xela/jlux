from transformers import T5Tokenizer, T5EncoderModel
import torch
import jax.numpy as jnp

class T5Wrapper:
    def __init__(self, path = "google/t5-v1_1-xxl" ):
        self._path = path
        self.device = "cuda"
        self._tokenizer = T5Tokenizer.from_pretrained(self._path)
        self._text_model = T5EncoderModel.from_pretrained(self._path, torch_dtype = torch.bfloat16).to(self.device).eval()

    def _tokenize(self, text, max_length =512, padding ="max_length", truncation=True, return_tensors="pt"):
        toks = self._tokenizer(text, max_length=max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)
        return toks
    
    def __call__(self, prompts: list[str]):
        toks = self._tokenize(prompts).to(self.device)
        with torch.no_grad():
            out = self._text_model(**toks).last_hidden_state
            return jnp.asarray(out.float().cpu().numpy())
        