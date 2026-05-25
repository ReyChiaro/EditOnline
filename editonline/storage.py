from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path

from fastapi import UploadFile
from PIL import Image

from .config import Settings
from .schemas import ImageRef


class Storage:
    """Small file registry for uploaded images, masks, and generated outputs."""

    def __init__(self, settings: Settings):
        self.root = settings.storage_dir.resolve()
        self.uploads = self.root / "uploads"
        self.masks = self.root / "masks"
        self.outputs = self.root / "outputs"
        for directory in (self.uploads, self.masks, self.outputs):
            directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, image_id: str) -> Path:
        for directory in (self.uploads, self.masks, self.outputs):
            for path in directory.iterdir():
                if path.stem == image_id:
                    return path
        raise FileNotFoundError(f"Unknown image id: {image_id}")

    def ref_for(self, path: Path) -> ImageRef:
        return ImageRef(id=path.stem, url=f"/files/{path.relative_to(self.root).as_posix()}", filename=path.name)

    async def save_upload(self, upload: UploadFile, bucket: str = "uploads") -> ImageRef:
        suffix = Path(upload.filename or "").suffix.lower() or self._suffix_for(upload.content_type)
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            suffix = ".png"
        directory = getattr(self, bucket)
        path = directory / f"{uuid.uuid4().hex}{suffix}"
        path.write_bytes(await upload.read())
        return self.ref_for(path)

    def save_image(self, image: Image.Image, bucket: str, suffix: str = ".png") -> ImageRef:
        directory = getattr(self, bucket)
        path = directory / f"{uuid.uuid4().hex}{suffix}"
        image.save(path)
        return self.ref_for(path)

    def open_image(self, image_id: str) -> Image.Image:
        return Image.open(self.path_for(image_id))

    @staticmethod
    def _suffix_for(content_type: str | None) -> str:
        return mimetypes.guess_extension(content_type or "") or ".png"
