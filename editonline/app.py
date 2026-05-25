from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .schemas import EditRequest, EditResponse, ImageRef, SegmentRequest, SegmentResponse, StatusResponse
from .services.inference_service import EditModelService
from .services.mask_service import MaskGeneratorService
from .storage import Storage


@lru_cache
def get_storage() -> Storage:
    return Storage(get_settings())


@lru_cache
def get_edit_service() -> EditModelService:
    return EditModelService(get_settings())


@lru_cache
def get_mask_service() -> MaskGeneratorService:
    return MaskGeneratorService(get_settings())


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health", response_model=StatusResponse)
    async def health(current_settings: Settings = Depends(get_settings)) -> StatusResponse:
        return StatusResponse(
            ok=True,
            app_name=current_settings.app_name,
            providers={"edit": ["local", "api"], "mask": ["local", "api", "manual-canvas"]},
        )

    @app.get("/files/{bucket}/{filename}", include_in_schema=False)
    async def get_file(bucket: str, filename: str, storage: Storage = Depends(get_storage)) -> FileResponse:
        path = (storage.root / bucket / filename).resolve()
        if storage.root not in path.parents or not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path)

    @app.post("/api/images", response_model=ImageRef)
    async def upload_image(file: UploadFile = File(...), storage: Storage = Depends(get_storage)) -> ImageRef:
        return await storage.save_upload(file, bucket="uploads")

    @app.post("/api/masks", response_model=ImageRef)
    async def upload_mask(file: UploadFile = File(...), storage: Storage = Depends(get_storage)) -> ImageRef:
        return await storage.save_upload(file, bucket="masks")

    @app.post("/api/segment", response_model=SegmentResponse)
    async def segment(
        request: SegmentRequest,
        storage: Storage = Depends(get_storage),
        service: MaskGeneratorService = Depends(get_mask_service),
    ) -> SegmentResponse:
        try:
            image = storage.open_image(request.image_id)
            result = await service.generate(request, image)
            refs = [storage.save_image(mask, "masks") for mask in result.masks]
            return SegmentResponse(masks=refs, provider=request.provider)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/edit", response_model=EditResponse)
    async def edit(
        request: EditRequest,
        storage: Storage = Depends(get_storage),
        service: EditModelService = Depends(get_edit_service),
    ) -> EditResponse:
        try:
            image = storage.open_image(request.image_id)
            mask = storage.open_image(request.mask_id) if request.mask_id else None
            result = await service.edit(request, image, mask)
            output_ref = storage.save_image(result.output, "outputs")
            visualization_ref = storage.save_image(result.visualization, "outputs") if result.visualization else None
            return EditResponse(output=output_ref, visualization=visualization_ref, provider=request.provider)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
