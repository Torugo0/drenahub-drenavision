"""Operational DrenaVision dashboard backed by the local FastAPI service."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import httpx
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from drenavision.core.config import get_settings
from drenavision.core.constants import DISCLAIMER, PUBLIC_CLASS_LABELS
from drenavision.core.image_io import InvalidImageError, decode_image

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = get_settings()
API_HOST = "127.0.0.1" if SETTINGS.api_host == "0.0.0.0" else SETTINGS.api_host
API_URL = os.getenv(
    "DRENAVISION_API_URL", f"http://{API_HOST}:{SETTINGS.api_port}"
).rstrip("/")
MODEL_PATH = SETTINGS.model_path
DATABASE_PATH = SETTINGS.database_path
EVIDENCE_DIR = SETTINGS.evidence_dir
DEMO_VIDEO_PATH = ROOT / "artifacts/videos/demo_input_test_sequence.mp4"
COLORS = {
    "obstrucao_visivel": "#ff6b35",
    "agua_acumulada": "#2d9cdb",
    "animal_visivel": "#8f5bd7",
}


def api_request(method: str, path: str, **kwargs) -> httpx.Response:
    try:
        return httpx.request(method, f"{API_URL}{path}", timeout=90, **kwargs)
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"API indisponível em {API_URL}. Inicie o serviço FastAPI."
        ) from exc


def render_video_preview(upload: BinaryIO, mime_type: str | None = None) -> None:
    """Render a file-like upload without passing Streamlit a memoryview."""

    video_format = (
        mime_type if mime_type and mime_type.startswith("video/") else "video/mp4"
    )
    upload.seek(0)
    st.video(
        upload,
        format=video_format,
        alt="Prévia do vídeo enviado para análise",
    )
    upload.seek(0)


def annotate(image_bytes: bytes, detections: list[dict]) -> Image.Image:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for detection in detections:
        name = detection["class_name"]
        x1, y1, x2, y2 = detection["bounding_box"]
        color = COLORS.get(name, "#ffffff")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = f"{PUBLIC_CLASS_LABELS.get(name, name)} {detection['confidence']:.0%}"
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        text_y = max(0, y1 - (bottom - top) - 8)
        draw.rectangle(
            (x1, text_y, x1 + (right - left) + 8, text_y + (bottom - top) + 8),
            fill=color,
        )
        draw.text((x1 + 4, text_y + 4), label, fill="white", font=font)
    return image


def render_prediction(image_bytes: bytes, payload: dict) -> None:
    detections = payload["detections"]
    st.image(
        annotate(image_bytes, detections),
        caption=f"Evento {payload['event_id']}",
        width="stretch",
    )
    if detections:
        st.warning(f"Revisão recomendada: {len(detections)} ocorrência(s) visível(is).")
        rows = [
            {
                "classe": PUBLIC_CLASS_LABELS.get(
                    item["class_name"], item["class_name"]
                ),
                "confiança": round(item["confidence"], 4),
                "bbox_xyxy_px": item["bounding_box"],
                "track_id": item.get("track_id"),
            }
            for item in detections
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.success("Nenhum objeto-alvo foi detectado com o limiar configurado.")
    st.caption(
        f"Modelo: {payload['model_version']} · inferência: "
        f"{payload['inference_ms']:.1f} ms · bbox: xyxy em pixels"
    )


def image_tab() -> None:
    st.subheader("Analisar imagem")
    source = st.radio(
        "Origem",
        ["Arquivo", "Câmera"],
        horizontal=True,
        label_visibility="collapsed",
    )
    media = (
        st.file_uploader("JPEG ou PNG", type=["jpg", "jpeg", "png"])
        if source == "Arquivo"
        else st.camera_input("Fotografe o ponto de drenagem")
    )
    point_id = st.text_input(
        "Identificador do ponto (opcional)",
        max_chars=80,
        placeholder="Ex.: bueiro-av-paulista-001",
    )
    if media is None:
        st.info("Envie uma imagem ou use a câmera para iniciar a triagem.")
        return
    if media.size > SETTINGS.max_image_bytes:
        limit_mib = SETTINGS.max_image_bytes / (1024 * 1024)
        st.error(f"A imagem excede o limite configurado de {limit_mib:.0f} MiB.")
        return
    image_bytes = media.getvalue()
    try:
        decode_image(image_bytes, max_pixels=SETTINGS.max_image_pixels)
    except InvalidImageError as exc:
        st.error(f"Imagem inválida: {exc}")
        return
    st.image(image_bytes, caption="Entrada", width="stretch")
    if st.button("Executar triagem", type="primary", use_container_width=True):
        with st.spinner("Executando o modelo customizado…"):
            try:
                response = api_request(
                    "POST",
                    "/vision/predict",
                    files={
                        "file": (
                            media.name,
                            image_bytes,
                            media.type or "image/jpeg",
                        )
                    },
                    data={"point_id": point_id},
                )
                if response.status_code != 200:
                    detail = response.json().get("detail", response.text)
                    st.error(f"Falha na análise ({response.status_code}): {detail}")
                    return
                render_prediction(image_bytes, response.json())
            except RuntimeError as exc:
                st.error(str(exc))


def video_tab() -> None:
    st.subheader("Processar vídeo")
    st.caption(
        "O processamento ocorre localmente, quadro a quadro, com rastreamento "
        "opcional. O arquivo original temporário é removido ao final."
    )
    if DEMO_VIDEO_PATH.is_file():
        st.download_button(
            "Baixar vídeo de teste (entrada sem anotações)",
            data=DEMO_VIDEO_PATH.read_bytes(),
            file_name=DEMO_VIDEO_PATH.name,
            mime="video/mp4",
            use_container_width=True,
        )
    upload = st.file_uploader(
        "Vídeo MP4, MOV, AVI ou MKV",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload",
    )
    track = st.checkbox("Ativar rastreamento entre quadros", value=True)
    point_id = st.text_input(
        "Identificador do ponto (opcional)",
        max_chars=80,
        placeholder="Ex.: bueiro-av-paulista-001",
        key="video_point",
    )
    if upload is None:
        return
    if any(marker in upload.name.lower() for marker in ("annotated", "anotado")):
        st.warning(
            "Este nome parece ser de um vídeo já processado. Para o teste padrão, "
            "use demo_input_test_sequence.mp4; o arquivo annotated é a saída."
        )
    if upload.size > SETTINGS.max_video_bytes:
        limit_mib = SETTINGS.max_video_bytes / (1024 * 1024)
        st.error(f"O vídeo excede o limite configurado de {limit_mib:.0f} MiB.")
        return
    render_video_preview(upload, upload.type)
    if not st.button("Processar vídeo", type="primary", use_container_width=True):
        return
    if not MODEL_PATH.is_file():
        st.error(
            "Checkpoint customizado ausente. Execute o treinamento ou configure "
            "DRENAVISION_MODEL_PATH."
        )
        return

    temporary_dir = ROOT / "artifacts/tmp"
    output_dir = ROOT / "artifacts/videos"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = Path(upload.name).suffix.lower()
    token = uuid4().hex
    input_path = temporary_dir / f"{token}{safe_suffix}"
    output_path = output_dir / f"{token}.mp4"
    upload.seek(0)
    with input_path.open("wb") as destination:
        shutil.copyfileobj(upload, destination, length=1024 * 1024)
    upload.seek(0)
    command = [
        sys.executable,
        str(ROOT / "scripts/process_video.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        str(MODEL_PATH),
        "--database",
        str(DATABASE_PATH),
        "--evidence-dir",
        str(EVIDENCE_DIR),
        "--conf",
        str(SETTINGS.confidence_threshold),
        "--iou",
        str(SETTINGS.iou_threshold),
        "--device",
        SETTINGS.device,
    ]
    if point_id.strip():
        command.extend(["--point-id", point_id.strip()])
    if track:
        command.append("--track")
    try:
        with st.spinner("Processando vídeo; isso pode levar alguns minutos…"):
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=7200,
                check=False,
            )
        if completed.returncode != 0 or not output_path.is_file():
            st.error("O processamento falhou.")
            diagnostic = completed.stderr[-4000:] or completed.stdout[-4000:]
            for private_path in (ROOT, MODEL_PATH, DATABASE_PATH, EVIDENCE_DIR):
                diagnostic = diagnostic.replace(str(private_path), "<caminho-local>")
            st.code(diagnostic)
            return
        st.success("Vídeo processado e evento registrado com sucesso.")
        st.video(str(output_path))
        sidecar_path = output_path.with_suffix(".json")
        if sidecar_path.is_file():
            summary = json.loads(sidecar_path.read_text(encoding="utf-8"))
            count_rows = [
                {
                    "classe": PUBLIC_CLASS_LABELS.get(class_name, class_name),
                    "contagem": details["count"],
                    "método": details["method"],
                }
                for class_name, details in summary.get("counts", {}).items()
            ]
            st.markdown("#### Resumo do vídeo")
            st.dataframe(count_rows, width="stretch", hide_index=True)
            video_info = summary.get("video", {})
            st.caption(
                f"{video_info.get('processed_frames', 0)} quadros · "
                f"{(video_info.get('processing_fps') or 0.0):.1f} FPS de processamento · "
                f"tracking: {'ativo' if track else 'inativo'}"
            )
            st.download_button(
                "Baixar manifesto JSON",
                data=sidecar_path.read_bytes(),
                file_name="drenavision_video_manifest.json",
                mime="application/json",
                use_container_width=True,
            )
        st.download_button(
            "Baixar vídeo anotado",
            data=output_path.read_bytes(),
            file_name="drenavision_anotado.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    except subprocess.TimeoutExpired:
        st.error("O processamento excedeu o limite de duas horas.")
    finally:
        input_path.unlink(missing_ok=True)


def history_tab() -> None:
    st.subheader("Histórico de triagens")
    point_id = st.text_input("Filtrar por ponto", key="history_point")
    if st.button("Atualizar histórico", use_container_width=True):
        try:
            params = {"limit": 100}
            if point_id.strip():
                params["point_id"] = point_id.strip()
            response = api_request("GET", "/vision/events", params=params)
            if response.status_code != 200:
                st.error(f"Falha ao consultar eventos: {response.text}")
                return
            payload = response.json()
            st.metric("Eventos encontrados", payload["total"])
            st.dataframe(payload["items"], width="stretch", hide_index=True)
        except RuntimeError as exc:
            st.error(str(exc))


def about_tab() -> None:
    st.subheader("Sobre o protótipo")
    st.markdown(
        """
O DrenaVision é um protótipo acadêmico de visão computacional para apoiar a
triagem visual de pontos de drenagem. O modelo procura três classes:
**obstrução visível**, **água acumulada** e **animal visível**.

As ocorrências são sinais para priorização operacional, não diagnósticos. O
histórico guarda metadados e evidência derivada anotada; a imagem original
enviada à API não é persistida.
"""
    )


def main() -> None:
    st.set_page_config(
        page_title="DrenaVision",
        page_icon="DV",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("DrenaVision")
    st.markdown("**Triagem visual assistida para drenagem urbana**")
    st.caption(DISCLAIMER)
    tabs = st.tabs(["Imagem", "Vídeo", "Histórico", "Sobre"])
    with tabs[0]:
        image_tab()
    with tabs[1]:
        video_tab()
    with tabs[2]:
        history_tab()
    with tabs[3]:
        about_tab()


if __name__ == "__main__":
    main()
