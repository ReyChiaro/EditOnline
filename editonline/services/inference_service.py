from __future__ import annotations

from dataclasses import dataclass

import httpx
import torch
from io import BytesIO
from PIL import Image

from editonline.config import Settings
from editonline.schemas import EditRequest, Provider


@dataclass(frozen=True)
class InferenceResult:
    output: Image.Image
    visualization: Image.Image | None


class EditModelService:
    """Provider switchboard for image editing models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._local_pipeline = None
        self._local_key: tuple[str, str, str, str] | None = None

    async def edit(self, request: EditRequest, image: Image.Image, mask: Image.Image | None) -> InferenceResult:
        if request.provider == Provider.api:
            return await self._edit_with_api(request, image, mask)
        return self._edit_with_local_model(request, image, mask)

    def _edit_with_local_model(self, request: EditRequest, image: Image.Image, mask: Image.Image | None) -> InferenceResult:
        base_model = request.base_model or self.settings.default_base_model
        lora_model = request.lora_model or self.settings.default_lora_model
        adapter_name = request.lora_adapter_name or self.settings.default_lora_adapter_name
        device = request.device or self.settings.default_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Local CUDA inference was requested, but CUDA is not available on this machine.")

        from inference.inference import load_pipeline, run_inference

        key = (base_model, lora_model, adapter_name, device)
        if self._local_pipeline is None or self._local_key != key:
            self._local_pipeline = load_pipeline(base_model, lora_model, adapter_name, device)
            self._local_key = key

        edit_mask = mask or Image.new("RGB", image.size, "white")
        with torch.inference_mode():
            output, visualization = run_inference(
                pipeline=self._local_pipeline,
                source_image=image,
                mask_image=edit_mask,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                enable_mask_to_box=request.enable_mask_to_box,
                mask_box_margin=request.mask_box_margin,
                enable_mask_blur=request.enable_mask_blur,
                blur_kernel=_odd(request.blur_kernel),
                blur_sigma=request.blur_sigma,
                enable_mask_dilation=request.enable_mask_dilation,
                dilation_kernel=_odd(request.dilation_kernel),
                num_inference_steps=request.num_inference_steps,
            )
        return InferenceResult(output=output, visualization=visualization)

    async def _edit_with_api(self, request: EditRequest, image: Image.Image, mask: Image.Image | None) -> InferenceResult:
        if not self.settings.edit_api_base_url:
            raise RuntimeError("EDITONLINE_EDIT_API_BASE_URL is required for API inference.")

        files = {
            "image": ("image.png", _image_bytes(image), "image/png"),
        }
        if mask is not None:
            files["mask"] = ("mask.png", _image_bytes(mask), "image/png")
        data = {"request": request.model_dump_json()}
        headers = _auth_headers(self.settings.edit_api_key)

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(self.settings.edit_api_base_url.rstrip("/") + "/edit", data=data, files=files, headers=headers)
            response.raise_for_status()
            output = Image.open(BytesIO(response.content)).convert("RGB")
        return InferenceResult(output=output, visualization=None)


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}
