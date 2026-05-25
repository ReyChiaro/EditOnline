from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class Provider(StrEnum):
    local = "local"
    api = "api"


class PointLabel(StrEnum):
    foreground = "foreground"
    background = "background"


class SegmentPoint(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: PointLabel = PointLabel.foreground


class SegmentBox(BaseModel):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)


class ImageRef(BaseModel):
    id: str
    url: str
    filename: str


class EditRequest(BaseModel):
    provider: Provider = Provider.local
    image_id: str
    mask_id: str | None = None
    prompt: str
    negative_prompt: str | None = None
    base_model: str | None = None
    lora_model: str | None = None
    lora_adapter_name: str | None = None
    device: str | None = None
    enable_mask_to_box: bool = False
    mask_box_margin: int = Field(default=200, ge=0, le=2048)
    enable_mask_blur: bool = False
    blur_kernel: int = Field(default=75, ge=1, le=501)
    blur_sigma: float = Field(default=15.0, ge=0.1, le=200)
    enable_mask_dilation: bool = False
    dilation_kernel: int = Field(default=25, ge=1, le=501)
    num_inference_steps: int = Field(default=50, ge=1, le=200)


class EditResponse(BaseModel):
    output: ImageRef
    visualization: ImageRef | None = None
    provider: Provider


class SegmentRequest(BaseModel):
    provider: Provider = Provider.local
    image_id: str
    points: list[SegmentPoint] = Field(default_factory=list)
    box: SegmentBox | None = None
    multimask_output: bool = True


class SegmentResponse(BaseModel):
    masks: list[ImageRef]
    provider: Provider


class StatusResponse(BaseModel):
    ok: bool
    app_name: str
    providers: dict[str, list[str]]


class RemoteEditPayload(BaseModel):
    image_url: HttpUrl
    mask_url: HttpUrl | None = None
    prompt: str
    negative_prompt: str | None = None

