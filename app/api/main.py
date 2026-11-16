"""DrenaVision HTTP API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.core.config import Settings, get_settings
from app.core.constants import (
    CLASS_NAMES,
    MODEL_UNAVAILABLE_DETAIL,
    PUBLIC_CLASS_LABELS,
    SUPPORTED_IMAGE_TYPES,
)
from app.core.image_io import InvalidImageError, decode_image
from app.core.schemas import (
    ClassDefinition,
    ClassesResponse,
    ErrorResponse,
    EventListResponse,
    HealthResponse,
    PredictionEvent,
)
from app.core.service import PredictionService, Predictor
from app.core.storage import EventRepository, EvidenceStore

LOGGER = logging.getLogger(__name__)


def _build_predictor(settings: Settings) -> Predictor | None:
    try:
        from app.inference.predictor import ModelUnavailableError, YoloPredictor
    except ImportError as exc:
        LOGGER.warning("Prediction runtime unavailable: %s", exc)
        return None
    try:
        return YoloPredictor(
            model_path=settings.model_path,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            device=settings.device,
        )
    except (FileNotFoundError, ModelUnavailableError) as exc:
        LOGGER.warning("Prediction model unavailable: %s", exc)
        return None
    except Exception:
        LOGGER.exception("Prediction model could not be initialized")
        return None


def create_app(
    *,
    settings: Settings | None = None,
    predictor: Predictor | None = None,
    repository: EventRepository | None = None,
) -> FastAPI:
    runtime = settings or get_settings()
    events = repository or EventRepository(runtime.database_path)
    supplied_predictor = predictor

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        events.initialize()
        active_predictor = supplied_predictor or _build_predictor(runtime)
        evidence_store = EvidenceStore(runtime.evidence_dir)
        app.state.predictor = active_predictor
        app.state.repository = events
        app.state.evidence_store = evidence_store
        app.state.inference_semaphore = asyncio.Semaphore(1)
        app.state.service = (
            PredictionService(
                active_predictor,
                events,
                evidence_store,
            )
            if active_predictor is not None
            else None
        )
        yield

    app = FastAPI(
        title="DrenaVision API",
        version=__version__,
        description=(
            "Triagem assistida de obstrução, água acumulada e animais visíveis "
            "em imagens de infraestrutura de drenagem."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        model = getattr(request.app.state, "predictor", None)
        loaded = bool(model is not None and getattr(model, "is_ready", True))
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_path=runtime.model_path.name,
            database_ready=runtime.database_path.parent.exists(),
            version=__version__,
        )

    @app.get("/vision/classes", response_model=ClassesResponse, tags=["metadata"])
    @app.get(
        "/v1/classes",
        response_model=ClassesResponse,
        include_in_schema=False,
    )
    def classes() -> ClassesResponse:
        return ClassesResponse(
            classes=[
                ClassDefinition(
                    class_id=class_id,
                    name=name,
                    label_pt_br=PUBLIC_CLASS_LABELS[name],
                )
                for class_id, name in CLASS_NAMES.items()
            ]
        )

    @app.post(
        "/vision/predict",
        response_model=PredictionEvent,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["prediction"],
    )
    @app.post(
        "/v1/predict/image",
        response_model=PredictionEvent,
        include_in_schema=False,
    )
    async def predict_image(
        request: Request,
        file: Annotated[UploadFile, File()],
        point_id: Annotated[str | None, Form(max_length=80)] = None,
    ) -> PredictionEvent:
        service: PredictionService | None = getattr(request.app.state, "service", None)
        if service is None or not getattr(service.predictor, "is_ready", True):
            raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL)
        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415, detail="Formato inválido; envie JPEG ou PNG."
            )
        data = await file.read(runtime.max_image_bytes + 1)
        await file.close()
        if len(data) > runtime.max_image_bytes:
            raise HTTPException(
                status_code=413, detail="Imagem excede o limite configurado."
            )
        try:
            image = decode_image(data, max_pixels=runtime.max_image_pixels)
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        normalized_point = point_id.strip() if point_id else None
        if point_id is not None and not normalized_point:
            normalized_point = None
        async with request.app.state.inference_semaphore:
            return await run_in_threadpool(
                partial(service.predict_image, image, point_id=normalized_point)
            )

    @app.get("/vision/events", response_model=EventListResponse, tags=["events"])
    @app.get("/v1/events", response_model=EventListResponse, include_in_schema=False)
    def list_events(
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        point_id: str | None = Query(default=None, max_length=80),
    ) -> EventListResponse:
        repo: EventRepository = request.app.state.repository
        return repo.list(limit=limit, offset=offset, point_id=point_id)

    @app.get(
        "/vision/events/{event_id}",
        response_model=PredictionEvent,
        responses={404: {"model": ErrorResponse}},
        tags=["events"],
    )
    @app.get(
        "/v1/events/{event_id}",
        response_model=PredictionEvent,
        include_in_schema=False,
    )
    def get_event(request: Request, event_id: str) -> PredictionEvent:
        repo: EventRepository = request.app.state.repository
        event = repo.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Evento não encontrado.")
        return event

    @app.get(
        "/vision/evidence/{event_id}",
        response_class=FileResponse,
        responses={404: {"model": ErrorResponse}},
        tags=["events"],
    )
    def get_evidence(request: Request, event_id: str) -> FileResponse:
        repo: EventRepository = request.app.state.repository
        event = repo.get(event_id)
        if event is None or event.evidence_uri is None:
            raise HTTPException(status_code=404, detail="Evidência não encontrada.")
        store: EvidenceStore = request.app.state.evidence_store
        try:
            evidence_path = store.resolve(event.evidence_uri)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Evidência inválida.") from exc
        if not evidence_path.is_file():
            raise HTTPException(status_code=404, detail="Evidência não encontrada.")
        media_type = (
            "application/json" if evidence_path.suffix == ".json" else "image/jpeg"
        )
        return FileResponse(evidence_path, media_type=media_type)

    return app


app = create_app()
