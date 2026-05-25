from __future__ import annotations

import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import torch
from PIL import Image

from editonline.config import Settings
from editonline.schemas import Provider, SegmentRequest


@dataclass(frozen=True)
class MaskResult:
    masks: list[Image.Image]


class MaskGeneratorService:
    """Provider switchboard for segmentation models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._sam = None
        self._sam_key: tuple[str, str, str] | None = None

    async def generate(self, request: SegmentRequest, image: Image.Image) -> MaskResult:
        if request.provider == Provider.api:
            return await self._generate_with_api(request, image)
        return self._generate_with_sam(request, image)

    def _generate_with_sam(self, request: SegmentRequest, image: Image.Image) -> MaskResult:
        self._ensure_sam_loaded()
        if self.settings.sam_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Local CUDA segmentation was requested, but CUDA is not available on this machine.")

        rgb = np.array(image.convert("RGB"))
        if request.points or request.box:
            from segment_anything import SamPredictor

            predictor = SamPredictor(self._sam)
            predictor.set_image(rgb)
            point_coords = np.array([[p.x, p.y] for p in request.points], dtype=np.float32) if request.points else None
            point_labels = (
                np.array([1 if p.label == "foreground" else 0 for p in request.points], dtype=np.int32)
                if request.points
                else None
            )
            box = None
            if request.box:
                box = np.array([request.box.x1, request.box.y1, request.box.x2, request.box.y2], dtype=np.float32)
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=request.multimask_output,
            )
            return MaskResult(masks=[_mask_to_image(mask) for mask in masks])

        from segment_anything import SamAutomaticMaskGenerator

        generator = SamAutomaticMaskGenerator(self._sam)
        masks = generator.generate(rgb)
        masks = sorted(masks, key=lambda item: item.get("area", 0), reverse=True)[:6]
        return MaskResult(masks=[_mask_to_image(item["segmentation"]) for item in masks])

    def _ensure_sam_loaded(self) -> None:
        model_type = self.settings.sam_model_type
        checkpoint = str(self.settings.sam_checkpoint)
        device = self.settings.sam_device
        key = (model_type, checkpoint, device)
        if self._sam is not None and self._sam_key == key:
            return

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Local CUDA segmentation was requested, but CUDA is not available on this machine.")
        sam_path = Path("mask_generator/model_hub/segment-anything").resolve()
        if str(sam_path) not in sys.path:
            sys.path.insert(0, str(sam_path))
        from segment_anything import sam_model_registry

        self._sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device=device)
        self._sam_key = key

    async def _generate_with_api(self, request: SegmentRequest, image: Image.Image) -> MaskResult:
        if not self.settings.mask_api_base_url:
            raise RuntimeError("EDITONLINE_MASK_API_BASE_URL is required for API mask generation.")

        files = {"image": ("image.png", _image_bytes(image), "image/png")}
        data = {"request": request.model_dump_json()}
        headers = {"Authorization": f"Bearer {self.settings.mask_api_key}"} if self.settings.mask_api_key else {}
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                self.settings.mask_api_base_url.rstrip("/") + "/segment",
                data=data,
                files=files,
                headers=headers,
            )
            response.raise_for_status()
            mask = Image.open(BytesIO(response.content)).convert("L")
        return MaskResult(masks=[mask])


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")


def _image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
