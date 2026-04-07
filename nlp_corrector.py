"""
Lightweight NLP correction module for notebook and offline evaluation use.
This module intentionally avoids MediaPipe/Streamlit dependencies.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

from transformers import AutoTokenizer, EncoderDecoderModel


class NLPCorrector(ABC):
    """Abstract base class for NLP-based text correction."""

    @abstractmethod
    def correct(self, text: str) -> str:
        """Apply correction to the input text."""

    @abstractmethod
    def get_suggestions(self, text: str) -> List[str]:
        """Get correction suggestions for the input text."""


@lru_cache(maxsize=1)
def load_indobert_corrector_model():
    """
    Load IndoBERT Seq2Seq model and tokenizer once per process.

    Returns:
        Tuple of (model, tokenizer, device)
    """
    import torch

    model_repo = "ZerXXX/indobert-corrector"
    subfolder = "indoBERT-best-corrector"

    tokenizer = AutoTokenizer.from_pretrained(model_repo, subfolder=subfolder)
    model = EncoderDecoderModel.from_pretrained(model_repo, subfolder=subfolder)

    # Explicit IDs from model config on Hugging Face.
    model.config.decoder_start_token_id = 2
    model.config.eos_token_id = 3
    model.config.pad_token_id = 0
    model.config.bos_token_id = 2

    if model.generation_config is not None:
        model.generation_config.decoder_start_token_id = 2
        model.generation_config.eos_token_id = 3
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 2

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, tokenizer, device


class IndoBERTCorrector(NLPCorrector):
    """
    IndoBERT Seq2Seq text corrector using EncoderDecoderModel.
    """

    def __init__(self):
        self.model, self.tokenizer, self.device = load_indobert_corrector_model()
        self.max_length = 64
        self.num_beams = 4

    def correct(self, text: str) -> str:
        if not text or not text.strip():
            return text

        try:
            import torch

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    do_sample=False,
                    early_stopping=True,
                    decoder_start_token_id=2,
                    eos_token_id=3,
                    pad_token_id=0,
                    bos_token_id=2,
                )

            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected if corrected else text

        except Exception as e:
            print(f"IndoBERT correction error: {e}")
            return text

    def get_suggestions(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        try:
            import torch

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    num_return_sequences=min(self.num_beams, 3),
                    do_sample=False,
                    early_stopping=True,
                    decoder_start_token_id=2,
                    eos_token_id=3,
                    pad_token_id=0,
                    bos_token_id=2,
                )

            suggestions = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                if decoded and decoded != text and decoded not in suggestions:
                    suggestions.append(decoded)

            return suggestions

        except Exception as e:
            print(f"IndoBERT suggestions error: {e}")
            return []
