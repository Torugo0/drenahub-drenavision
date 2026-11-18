"""DrenaVision HTTP API."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from drenavision import __version__
from drenavision.core.config import Settings, get_settings
from drenavision.core.constants import (
    CLASS_NAMES,
    MODEL_UNAVAILABLE_DETAIL,
    PUBLIC_CLASS_LABELS,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
)
from drenavision.core.image_io import InvalidImageError, decode_image
from drenavision.core.schemas import (
    ClassDefinition,
    ClassesResponse,
    ErrorResponse,
    EventListResponse,
    HealthResponse,
    PredictionEvent,
)
from drenavision.core.service import PredictionService, Predictor
from drenavision.core.storage import EventRepository, EvidenceStore
from drenavision.inference.errors import InferenceError
from drenavision.inference.utils import write_json
from drenavision.inference.video import persist_video_event, process_video

LOGGER = logging.getLogger(__name__)
VIDEO_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


def _video_artifacts(store: EvidenceStore, event_id: str) -> list[Path]:
    """Only derived names owned by this new canonical UUID; never user inputs."""
    names = [f"videos/{event_id}{suffix}" for suffix in (".mp4", ".json", ".jpg")]
    names.extend(f"videos/{event_id}_{name}.jpg" for name in CLASS_NAMES.values())
    names.extend((f"{event_id}.json", f"{event_id}.json.part"))
    return [store.resolve(name) for name in names]


def _rollback_video(repo: EventRepository, store: EvidenceStore, event_id: str) -> None:
    # The UUID was checked absent before processing. A failed repository adapter
    # could commit then raise; remove only this newly-created video event.
    with sqlite3.connect(repo.database_path, timeout=10) as connection:
        connection.execute(
            "DELETE FROM prediction_events WHERE event_id = ? AND input_type = 'video'",
            (event_id,),
        )
    for path in _video_artifacts(store, event_id):
        path.unlink(missing_ok=True)


def _build_predictor(settings: Settings) -> Predictor | None:
    try:
        from drenavision.inference.predictor import ModelUnavailableError
        from drenavision.inference.runtime import create_predictor
    except ImportError as exc:
        LOGGER.warning("Prediction runtime unavailable: %s", exc)
        return None
    try:
        return create_predictor(
            model_path=settings.model_path,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            device=settings.device,
            image_size=settings.image_size,
            water_model_path=settings.water_model_path,
            water_pixel_confidence=settings.water_pixel_confidence,
            water_min_area_fraction=settings.water_min_area_fraction,
            water_mode=settings.water_mode,
            residue_confidence=settings.confidence_threshold,
            animal_confidence=settings.animal_confidence_threshold,
            object_model_path=settings.object_model_path,
            object_verifier_path=settings.object_verifier_path,
            object_residue_min_margin=settings.object_residue_min_margin,
            object_animal_min_margin=settings.object_animal_min_margin,
            object_residue_confidence=settings.object_residue_confidence,
            object_animal_confidence=settings.object_animal_confidence,
            object_text_confidence=settings.object_text_confidence,
            object_mode=settings.object_mode,
            object_shortest_edge=settings.object_shortest_edge,
            object_longest_edge=settings.object_longest_edge,
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
            "Triagem assistida de resíduos, água visível e possíveis animais "
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

    @app.post(
        "/vision/predict/video",
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["prediction"],
    )
    async def predict_video(
        request: Request,
        file: Annotated[UploadFile, File()],
        point_id: Annotated[str | None, Form(max_length=80)] = None,
    ) -> dict:
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            await file.close()
            raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL)
        suffix = Path(file.filename or "").suffix.lower()
        if (
            suffix not in VIDEO_MIME_BY_SUFFIX
            or file.content_type not in SUPPORTED_VIDEO_TYPES
            or VIDEO_MIME_BY_SUFFIX.get(suffix) != file.content_type
        ):
            await file.close()
            raise HTTPException(
                status_code=415, detail="Formato inválido; envie MP4, MOV, AVI ou MKV."
            )
        repo: EventRepository = request.app.state.repository
        store: EvidenceStore = request.app.state.evidence_store
        event_id = str(uuid4())
        artifacts = _video_artifacts(store, event_id)
        if repo.get(event_id) is not None or any(path.exists() for path in artifacts):
            await file.close()
            raise HTTPException(
                status_code=409,
                detail="Não foi possível reservar o registro; tente novamente.",
            )
        destination = store.resolve(f"videos/{event_id}.mp4")
        normalized_point = point_id.strip() if point_id and point_id.strip() else None
        try:
            with tempfile.TemporaryDirectory(
                prefix="drenavision_upload_video_"
            ) as scratch:
                source = Path(scratch) / f"input{suffix}"
                size = 0
                with source.open("wb") as uploaded:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > runtime.max_video_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail="Vídeo excede o limite configurado.",
                            )
                        uploaded.write(chunk)
                if size == 0:
                    raise HTTPException(
                        status_code=400, detail="O vídeo enviado está vazio."
                    )
                async with request.app.state.inference_semaphore:
                    manifest = await run_in_threadpool(
                        partial(
                            process_video,
                            source,
                            destination,
                            runtime.model_path,
                            confidence=runtime.confidence_threshold,
                            iou=runtime.iou_threshold,
                            device=runtime.device,
                            image_size=runtime.image_size,
                            track=True,
                            water_model_path=runtime.water_model_path,
                            water_pixel_confidence=runtime.water_pixel_confidence,
                            water_min_area_fraction=runtime.water_min_area_fraction,
                            water_mode=runtime.water_mode,
                            residue_confidence=runtime.confidence_threshold,
                            animal_confidence=runtime.animal_confidence_threshold,
                            water_confirmation_seconds=runtime.video_water_confirmation_seconds,
                            object_model_path=runtime.object_model_path,
                            object_residue_confidence=runtime.object_residue_confidence,
                            object_animal_confidence=runtime.object_animal_confidence,
                            object_text_confidence=runtime.object_text_confidence,
                            object_mode=runtime.object_mode,
                            object_shortest_edge=runtime.object_shortest_edge,
                            object_longest_edge=runtime.object_longest_edge,
                            object_verifier_path=runtime.object_verifier_path,
                            object_residue_min_margin=runtime.object_residue_min_margin,
                            object_animal_min_margin=runtime.object_animal_min_margin,
                            shared_predictor=predictor,
                        )
                    )
                    if (
                        manifest.get("status") != "completed"
                        or not destination.is_file()
                    ):
                        raise RuntimeError("processador não produziu um vídeo completo")
                    manifest.update(
                        event_id=event_id, video_url=f"/vision/videos/{event_id}"
                    )
                    # Check serializability and write the sidecar before the final
                    # database commit, so a write failure cannot leave a stale row.
                    json.dumps(manifest, allow_nan=False)
                    write_json(destination.with_suffix(".json"), manifest)
                    event = await run_in_threadpool(
                        partial(
                            persist_video_event,
                            manifest,
                            repo,
                            store,
                            point_id=normalized_point,
                            event_id=event_id,
                        )
                    )
                    return {**manifest, "event": event.model_dump(mode="json")}
        except Exception as exc:
            await run_in_threadpool(partial(_rollback_video, repo, store, event_id))
            if isinstance(exc, HTTPException):
                raise
            LOGGER.exception("Video inspection failed for event %s", event_id)
            if isinstance(exc, InferenceError):
                raise HTTPException(
                    status_code=503, detail=MODEL_UNAVAILABLE_DETAIL
                ) from exc
            if isinstance(exc, (ValueError, RuntimeError)):
                raise HTTPException(
                    status_code=400,
                    detail="Não foi possível analisar este vídeo. Confira o arquivo e tente novamente.",
                ) from exc
            raise HTTPException(
                status_code=500, detail="Não foi possível concluir a análise do vídeo."
            ) from exc
        finally:
            await file.close()

    @app.get("/vision/videos/{event_id}", response_class=FileResponse, tags=["events"])
    def get_video(request: Request, event_id: str) -> FileResponse:
        try:
            identifier = str(UUID(event_id))
        except ValueError as exc:
            raise HTTPException(
                status_code=404, detail="Vídeo não encontrado."
            ) from exc
        event = request.app.state.repository.get(identifier)
        if event is None or event.input_type != "video":
            raise HTTPException(status_code=404, detail="Vídeo não encontrado.")
        store: EvidenceStore = request.app.state.evidence_store
        try:
            path = store.resolve(f"videos/{identifier}.mp4")
        except ValueError as exc:
            raise HTTPException(
                status_code=404, detail="Vídeo não encontrado."
            ) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Vídeo não encontrado.")
        return FileResponse(
            path, media_type="video/mp4", filename=f"drenavision_{identifier}.mp4"
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
