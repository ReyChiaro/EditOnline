import os
import copy
import torch
import argparse
import torch.nn.functional as F
import torchvision.transforms.functional as T

from PIL import Image
from pathlib import Path
from torchvision.utils import save_image
from torchvision.ops import masks_to_boxes
from .qwenimage_edit_plus.pipeline_qwenimage_mask_edit import QwenImageMaskEditPipeline


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def dilate_mask(mask: torch.Tensor, dilation_kernel_size: int) -> torch.Tensor:
    padding = dilation_kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=dilation_kernel_size, padding=padding, stride=1)
    return dilated


def mask_to_box(mask: torch.Tensor, margin: int) -> torch.Tensor:
    boxes = masks_to_boxes(mask.unsqueeze(0) if mask.ndim == 2 else mask)
    box = boxes[0].to(torch.int64)
    w_min, h_min, w_max, h_max = box

    w_min = max(0, w_min - margin)
    h_min = max(0, h_min - margin)
    w_max = min(mask.shape[-1], w_max + margin)
    h_max = min(mask.shape[-2], h_max + margin)

    regular_mask = torch.zeros_like(mask)
    if mask.ndim == 3:
        regular_mask[:, h_min : h_max + 1, w_min : w_max + 1] = 1
    else:
        regular_mask[h_min : h_max + 1, w_min : w_max + 1] = 1
    return regular_mask


# ---------------------------------------------------------------------------
# Public API (used by the FastAPI server)
# ---------------------------------------------------------------------------


def load_pipeline(
    base_model: str,
    lora_model: str,
    lora_adapter_name: str = "default",
    device: str = "cuda:0",
) -> QwenImageMaskEditPipeline:
    """Load the base model and attach the LoRA adapter, then move to *device*."""
    pipeline = QwenImageMaskEditPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    pipeline.transformer.load_lora_adapter(
        lora_model,
        prefix=None,
        adapter_name=lora_adapter_name,
    )
    pipeline.transformer.set_adapter(lora_adapter_name)
    return pipeline


def run_inference(
    pipeline: QwenImageMaskEditPipeline,
    source_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    negative_prompt: str | None = None,
    enable_mask_to_box: bool = False,
    mask_box_margin: int = 200,
    enable_mask_blur: bool = False,
    blur_kernel: int = 75,
    blur_sigma: float = 15.0,
    enable_mask_dilation: bool = False,
    dilation_kernel: int = 25,
    num_inference_steps: int = 50,
) -> tuple[Image.Image, Image.Image]:
    """Run a single inference pass.

    Returns
    -------
    output : PIL.Image
        The generated (inpainted) image at the original resolution.
    visualization : PIL.Image
        A 2-row × 3-column grid showing source / mask / output side by side
        with a colour-overlay row for quick visual inspection.
    """
    source_image = source_image.convert("RGB")
    mask_image = mask_image.convert("RGB")

    # -------- Mask Processing -------- #
    mask_tensor = T.to_tensor(mask_image)
    mask_tensor = (mask_tensor > 0.5).to(dtype=torch.float32)
    if enable_mask_to_box:
        mask_tensor = mask_to_box(mask_tensor, mask_box_margin)
    if enable_mask_dilation:
        mask_tensor = dilate_mask(mask_tensor, dilation_kernel)
    if enable_mask_blur:
        blur_mask_tensor = T.gaussian_blur(mask_tensor, kernel_size=blur_kernel, sigma=blur_sigma)
        blur_mask_tensor[mask_tensor < 0.5] = blur_mask_tensor[mask_tensor < 0.5] * 2
        blur_mask_tensor[mask_tensor > 0.5] = 1
        mask_tensor = blur_mask_tensor
    processed_mask = T.to_pil_image(mask_tensor)

    # -------- Generation -------- #
    output: Image.Image = pipeline(
        image=source_image,
        mask=processed_mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        output_type="pil",
        return_dict=True,
        unmask_with="noisy_source",
        enable_concat_mask=True,
    ).images[0]

    # -------- Visualization Grid -------- #
    H, W = source_image.height, source_image.width

    source_tensor = T.to_tensor(source_image)
    mask_vis = T.resize(T.to_tensor(processed_mask), [H, W])
    output_tensor = T.resize(T.to_tensor(output), [H, W])

    colored_mask = copy.deepcopy(mask_vis)
    colored_mask[1, ...] = 0
    colored_mask[2, ...] = 0
    masked_source = torch.where(mask_vis == 1, 0.4 * source_tensor + 0.6 * colored_mask, source_tensor)
    masked_predict = torch.where(mask_vis > 0, 0.4 * output_tensor + 0.6 * colored_mask, output_tensor)

    grid = torch.cat(
        [
            torch.cat([source_tensor, mask_vis, output_tensor], dim=-1),
            torch.cat([masked_source, mask_vis, masked_predict], dim=-1),
        ],
        dim=-2,
    )
    visualization = T.to_pil_image(grid.clamp(0, 1))

    return output, visualization


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")

    # -------- Pipeline and LoRA -------- #
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--lora-model", type=str, required=True)
    parser.add_argument("--lora-adapter-name", type=str, default="default")

    # -------------- Data --------------- #
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--source-image", type=str, required=True)
    parser.add_argument("--mask-image", type=str, required=True)
    parser.add_argument("--enable-mask-to-box", action="store_true", default=False)
    parser.add_argument("--mask-box-margin", type=int, default=200)
    parser.add_argument("--enable-mask-blur", action="store_true", default=False)
    parser.add_argument("--blur-kernel", type=int, default=75)
    parser.add_argument("--blur-sigma", type=float, default=15.0)
    parser.add_argument("--enable-mask-dilation", action="store_true", default=False)
    parser.add_argument("--dilation-kernel", type=int, default=25)

    # ------------ Inference ------------ #
    parser.add_argument("--save-dir", type=str, default="output")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    _pipeline = load_pipeline(args.base_model, args.lora_model, args.lora_adapter_name, args.device)

    _source = Image.open(args.source_image)
    _mask = Image.open(args.mask_image)
    image_name = Path(args.source_image).stem

    with torch.inference_mode():
        _output, _vis = run_inference(
            pipeline=_pipeline,
            source_image=_source,
            mask_image=_mask,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            enable_mask_to_box=args.enable_mask_to_box,
            mask_box_margin=args.mask_box_margin,
            enable_mask_blur=args.enable_mask_blur,
            blur_kernel=args.blur_kernel,
            blur_sigma=args.blur_sigma,
            enable_mask_dilation=args.enable_mask_dilation,
            dilation_kernel=args.dilation_kernel,
            num_inference_steps=args.num_inference_steps,
        )

    _vis_tensor = T.to_tensor(_vis)
    save_image(_vis_tensor, os.path.join(args.save_dir, f"output-{image_name}.jpg"))
