"""
SigLIP-based multimodal encoder for skincare product visual search.

Encodes both images and text into a shared 768-d embedding space
using google/siglip-base-patch16-224.
"""

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, SiglipModel

logger = logging.getLogger(__name__)


class SigLIPEngine:
    """Multimodal encoder using SigLIP for image↔text embedding."""

    MODEL_NAME = "google/siglip-base-patch16-224"
    EMBEDDING_DIM = 768

    def __init__(self, device: str | None = None):
        """Initialize the encoder (lazy-loads model on first call).

        Args:
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self._model: SiglipModel | None = None
        self._processor: Any = None
        logger.info("SigLIPEngine ready (device=%s, lazy loading)", self.device)

    # ── Private helpers ────────────────────────────────────────────

    def _load(self):
        """Lazy-load model & processor on first use."""
        if self._model is not None:
            return
        logger.info("Downloading / loading SigLIP model …")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        self._model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self._model.eval()  # type: ignore[union-attr]
        logger.info("SigLIP model loaded ✓")

    @staticmethod
    def _to_pil(img: Union[Image.Image, str, Path]) -> Image.Image:
        if isinstance(img, (str, Path)):
            return Image.open(img).convert("RGB")
        return img.convert("RGB")

    # ── Private: extract tensor ──────────────────────────────────

    @staticmethod
    def _as_tensor(out) -> torch.Tensor:
        """Handle both raw tensor and ModelOutput returns from transformers."""
        if isinstance(out, torch.Tensor):
            return out
        # transformers v5+ may return BaseModelOutputWithPooling
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]
        raise TypeError(f"Unexpected model output type: {type(out)}")

    def _norm(self, t: torch.Tensor) -> torch.Tensor:
        return t / t.norm(dim=-1, keepdim=True)

    # ── Public API ─────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, str, Path]) -> np.ndarray:
        """Encode a single image → normalized (768,) vector."""
        self._load()
        assert self._processor is not None and self._model is not None
        pil = self._to_pil(image)
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)  # type: ignore[misc]
        emb = self._as_tensor(self._model.get_image_features(**inputs))
        return self._norm(emb).cpu().numpy().flatten()

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        """Encode a batch of images → normalized (N, 768) matrix."""
        self._load()
        assert self._processor is not None and self._model is not None
        pils = [self._to_pil(i) for i in images]
        inputs = self._processor(
            images=pils, return_tensors="pt", padding=True
        ).to(self.device)  # type: ignore[misc]
        emb = self._as_tensor(self._model.get_image_features(**inputs))
        return self._norm(emb).cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text → normalized (768,) vector."""
        self._load()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(
            text=text, return_tensors="pt", padding=True
        ).to(self.device)  # type: ignore[misc]
        emb = self._as_tensor(self._model.get_text_features(**inputs))
        return self._norm(emb).cpu().numpy().flatten()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts → normalized (N, 768) matrix."""
        self._load()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)  # type: ignore[misc]
        emb = self._as_tensor(self._model.get_text_features(**inputs))
        return self._norm(emb).cpu().numpy()
