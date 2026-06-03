import jax.numpy as jnp
import torch
from transformers import T5EncoderModel, T5Tokenizer


class T5Wrapper:
    def __init__(self, dtype, path="google/t5-v1_1-xxl"):
        self._path = path
        self.dtype = dtype
        self.device = "cuda"
        self._tokenizer = T5Tokenizer.from_pretrained(self._path)
        self._text_model = (
            T5EncoderModel.from_pretrained(self._path, torch_dtype=torch.bfloat16)
            .to(self.device)
            .eval()
        )

    def _tokenize(
        self, text, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    ):
        toks = self._tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return toks

    def unload(self):
        self._text_model.to("cpu")
        torch.cuda.empty_cache()

    def __call__(self, prompts: list[str]):
        self._text_model.to(self.device)
        toks = self._tokenize(prompts).to(self.device)
        with torch.no_grad():
            out = self._text_model(**toks).last_hidden_state
            return jnp.asarray(out.float().cpu().numpy()).astype(self.dtype)
