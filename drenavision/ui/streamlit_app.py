"""Citizen-facing image and video inspection with persistent, visual results."""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import httpx
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

from drenavision.core.config import Settings, get_settings
from drenavision.core.image_io import InvalidImageError, decode_image
from drenavision.inference.video_codec import prepare_browser_video

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = get_settings()
API_HOST = "127.0.0.1" if SETTINGS.api_host == "0.0.0.0" else SETTINGS.api_host
API_URL = os.getenv(
    "DRENAVISION_API_URL", f"http://{API_HOST}:{SETTINGS.api_port}"
).rstrip("/")
EXAMPLES_DIR = ROOT / "dados/examples"
RESULTS_DIR = ROOT / "resultados/local"
LABELS = {
    "obstrucao_visivel": "Resíduos ou possível obstrução",
    "agua_acumulada": "Água acumulada",
    "animal_visivel": "Possível animal",
}
ADVICE = {
    "obstrucao_visivel": "Peça à equipe responsável para verificar os resíduos e a passagem da água.",
    "agua_acumulada": "Peça uma verificação do acúmulo de água e das condições de drenagem.",
    "animal_visivel": "Avise a equipe responsável para verificar a presença do animal no local.",
}
COLORS = {
    "obstrucao_visivel": "#EA580C",
    "agua_acumulada": "#0284C7",
    "animal_visivel": "#7C3AED",
}


def api_request(method: str, path: str, **kwargs) -> httpx.Response:
    try:
        return httpx.request(method, f"{API_URL}{path}", timeout=90, **kwargs)
    except httpx.RequestError as exc:
        raise RuntimeError(
            "Não foi possível conectar ao serviço de análise. Abra iniciar.ps1 e tente novamente."
        ) from exc


def annotate(image_bytes: bytes, detections: list[dict]) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_size = max(14, min(image.size) // 35)
    font = ImageFont.load_default(size=font_size)
    for font_name in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except OSError:
            continue
    line_width = max(3, min(image.size) // 180)
    for item in detections:
        x1, y1, x2, y2 = item["bounding_box"]
        color = COLORS.get(item["class_name"], "#0F766E")
        label = LABELS.get(item["class_name"], "Objeto identificado")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        box = draw.textbbox((0, 0), label, font=font)
        label_width = box[2] + 12
        label_height = box[3] - box[1] + 12
        left = max(0, min(x1, image.width - label_width))
        top = max(0, y1 - label_height)
        draw.rectangle((left, top, left + label_width, top + label_height), fill=color)
        draw.text((left + 6, top + 6 - box[1]), label, fill="white", font=font)
    return image


def result_counts(payload: dict, kind: str) -> dict[str, int]:
    """Count objects without adding the same object once for every video frame."""
    if kind == "video":
        return {
            name: int(payload.get("counts", {}).get(name, {}).get("count", 0))
            for name in LABELS
        }
    counts = Counter(item["class_name"] for item in payload.get("detections", []))
    return {name: counts[name] for name in LABELS}


def render_alerts(counts: dict[str, int], kind: str) -> None:
    found = [name for name, count in counts.items() if count > 0]
    if found:
        st.warning(
            "Atenção: encontramos sinais que precisam de uma verificação no local.",
            icon="⚠️",
        )
    else:
        st.info(
            "Nenhum sinal foi identificado nesta análise. Confira também a imagem ou o vídeo: a análise pode deixar passar objetos.",
            icon="🔎",
        )
    for column, (name, label) in zip(st.columns(3), LABELS.items(), strict=True):
        with column, st.container(border=True):
            value = "Não identificado"
            if counts[name]:
                value = (
                    f"{counts[name]} marcação"
                    if counts[name] == 1
                    else f"{counts[name]} marcações"
                )
            st.metric(label, value)
    if kind == "video" and found:
        st.caption(
            "As marcações resumem os sinais no vídeo. O mesmo objeto pode voltar a ser contado se sair de vista."
        )
    for name in found:
        st.markdown(f"**{LABELS[name]}:** {ADVICE[name]}")


def load_examples() -> list[dict]:
    manifest = EXAMPLES_DIR / "sources.json"
    if not manifest.is_file():
        return []
    items = [
        item
        for item in json.loads(manifest.read_text(encoding="utf-8"))["items"]
        if (EXAMPLES_DIR / item["path"]).is_file()
    ]
    return sorted(items, key=lambda item: item["id"] != "ralo_lixo")


def analyze_image(data: bytes, name: str, point: str) -> dict:
    decode_image(data, max_pixels=SETTINGS.max_image_pixels)
    mime = "image/png" if Path(name).suffix.lower() == ".png" else "image/jpeg"
    response = api_request(
        "POST",
        "/vision/predict",
        files={"file": (name, data, mime)},
        data={"point_id": point},
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Não foi possível analisar a foto. Confira se o modelo está disponível e tente novamente."
        )
    payload = response.json()
    return {
        "kind": "image",
        "payload": payload,
        "image": annotate(data, payload["detections"]),
    }


def analyze_video(source: Path, point: str) -> dict:
    # A Streamlit session may outlive configuration changes. Read and validate
    # the current configuration for every job instead of reusing module state.
    runtime = Settings()
    if not runtime.model_path.is_file():
        raise RuntimeError(
            "O modelo de análise não foi encontrado. Reinicie com iniciar.ps1 -Reiniciar. "
            "Se o aviso continuar, confira a instalação do modelo no README."
        )
    token = uuid4().hex
    destination = RESULTS_DIR / "videos" / f"{token}.mp4"
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(ROOT / "ferramentas/scripts/process_video.py"),
        "--input",
        str(source),
        "--output",
        str(destination),
        "--model",
        str(runtime.model_path),
        "--database",
        str(runtime.database_path),
        "--evidence-dir",
        str(runtime.evidence_dir),
        "--conf",
        str(runtime.confidence_threshold),
        "--iou",
        str(runtime.iou_threshold),
        "--device",
        runtime.device,
        "--track",
    ]
    if point:
        command.extend(["--point-id", point])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    sidecar = destination.with_suffix(".json")
    if completed.returncode != 0 or not destination.is_file() or not sidecar.is_file():
        log = RESULTS_DIR / "logs" / f"video_{token}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            json.dumps(
                {"command": command, "exit_code": completed.returncode},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
            + completed.stderr
            + completed.stdout,
            encoding="utf-8",
        )
        raise RuntimeError(
            "Não foi possível concluir este vídeo. Tente um clipe mais curto ou outro arquivo. O detalhe foi salvo para a equipe técnica."
        )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return {"kind": "video", "payload": payload, "path": str(destination)}


def render_result(result: dict) -> None:
    payload, kind = result["payload"], result["kind"]
    st.subheader("Resultado da análise")
    st.caption(
        f"Arquivo: {result['name']}"
        + (f" · Local: {result['point']}" if result["point"] else "")
    )
    render_alerts(result_counts(payload, kind), kind)
    if kind == "image":
        st.image(
            result["image"],
            caption="Os contornos mostram o que foi identificado.",
            width="stretch",
        )
        output = BytesIO()
        result["image"].save(output, format="JPEG")
        st.download_button(
            "Salvar foto com marcações",
            output.getvalue(),
            "drenavision_resultado.jpg",
            "image/jpeg",
            key="download_image",
            on_click="ignore",
        )
    else:
        path = Path(result["path"])
        poster = path.with_suffix(".jpg")
        if poster.is_file():
            st.image(
                str(poster),
                caption="Quadro de destaque: confira as marcações antes de reproduzir o vídeo.",
                width="stretch",
            )
        st.markdown("**Vídeo com as marcações — aperte ▶ para assistir**")
        st.video(str(path), format="video/mp4")
        for frame in payload.get("review_frames", []):
            at = float(frame.get("timestamp_seconds", 0))
            names = sorted(
                {
                    LABELS.get(item["class_name"], "Objeto")
                    for item in frame.get("detections", [])
                }
            )
            if names:
                st.caption(f"{at:.1f}s — {', '.join(names)}")
        st.download_button(
            "Salvar vídeo com marcações",
            path.read_bytes(),
            "drenavision_resultado.mp4",
            "video/mp4",
            key="download_video",
            on_click="ignore",
        )
    st.caption(
        "A análise ajuda a escolher o que verificar. A confirmação é feita por uma pessoa no local."
    )
    with st.expander("Detalhes para a equipe técnica"):
        st.json(payload, expanded=False)
        st.download_button(
            "Baixar relatório técnico",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "drenavision_relatorio.json",
            "application/json",
            key="download_report",
            on_click="ignore",
        )


def analysis_page() -> None:
    st.markdown("### Confira um ponto de drenagem")
    st.write(
        "Envie uma foto ou um vídeo. Vamos destacar possíveis resíduos, água acumulada e animais, e indicar o que precisa de atenção."
    )
    examples = load_examples()
    choice = st.radio(
        "Como deseja começar?",
        ["Enviar foto ou vídeo", "Experimentar um exemplo", "Tirar uma foto"],
        horizontal=True,
    )
    media = None
    source_path = None
    if choice == "Experimentar um exemplo":
        if not examples:
            st.info(
                "Os exemplos ainda não estão disponíveis. Confira a preparação em iniciar.ps1."
            )
            return
        example = st.selectbox(
            "Escolha uma situação", examples, format_func=lambda item: item["title"]
        )
        source_path = EXAMPLES_DIR / example["path"]
        st.caption(example["description"])
        thumb = EXAMPLES_DIR / example.get("thumbnail", "")
        if thumb.is_file():
            st.image(str(thumb), width=440)
        st.caption(
            f"{'Filmagem real' if example.get('is_recorded_video') else 'Fotografia'} · {example['author']} · {example['license']}"
        )
        st.markdown(f"[Ver origem e créditos]({example['source_url']})")
    elif choice == "Tirar uma foto":
        media = st.camera_input("Fotografe o local que deseja verificar")
    else:
        media = st.file_uploader(
            "Escolha uma foto ou um vídeo",
            type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"],
            key="inspection_upload",
            help="Fotos: até 12 MB. Vídeos: até 500 MB. Para começar, prefira clipes curtos.",
        )
    point = st.text_input(
        "Onde foi feito o registro? (opcional)",
        placeholder="Ex.: ralo da garagem, entrada do bloco A",
        max_chars=80,
    ).strip()
    if media is None and source_path is None:
        st.info(
            "Escolha um arquivo acima ou experimente um dos exemplos para ver como funciona."
        )
        return
    name = source_path.name if source_path else media.name
    kind = (
        "image" if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"} else "video"
    )
    size = source_path.stat().st_size if source_path else media.size
    limit = SETTINGS.max_image_bytes if kind == "image" else SETTINGS.max_video_bytes
    if size > limit:
        st.error(
            "Este arquivo é muito grande. Escolha uma foto menor ou um trecho mais curto do vídeo."
        )
        return
    data = source_path.read_bytes() if source_path else media.getvalue()
    identity = hashlib.sha256(data + name.encode() + point.encode()).hexdigest()
    if st.button("Analisar foto ou vídeo", type="primary", width="stretch"):
        st.session_state.pop("inspection_result", None)
        temporary = None
        try:
            with st.spinner("Analisando o registro e preparando as marcações…"):
                if kind == "image":
                    result = analyze_image(data, name, point)
                else:
                    if source_path is None:
                        temporary = (
                            RESULTS_DIR
                            / "tmp"
                            / f"{uuid4().hex}{Path(name).suffix.lower()}"
                        )
                        temporary.parent.mkdir(parents=True, exist_ok=True)
                        with temporary.open("wb") as target:
                            shutil.copyfileobj(BytesIO(data), target)
                    result = analyze_video(source_path or temporary, point)
                result.update({"identity": identity, "name": name, "point": point})
                st.session_state["inspection_result"] = result
        except (
            RuntimeError,
            InvalidImageError,
            OSError,
            ValueError,
            subprocess.TimeoutExpired,
        ) as exc:
            st.error(
                str(exc)
                if isinstance(exc, (RuntimeError, InvalidImageError))
                else "Não foi possível ler este arquivo. Tente outro registro."
            )
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    result = st.session_state.get("inspection_result")
    if result and result["identity"] == identity:
        render_result(result)
    with st.expander("Ver arquivo original"):
        if kind == "image":
            st.image(data, width="stretch")
        elif st.button("Preparar prévia do original"):
            cache = RESULTS_DIR / "tmp" / "previews"
            cache.mkdir(parents=True, exist_ok=True)
            original = cache / f"{identity}{Path(name).suffix.lower()}"
            preview = cache / f"{identity}_browser.mp4"
            try:
                if not preview.is_file():
                    original.write_bytes(data)
                    with st.spinner("Preparando vídeo para reprodução…"):
                        prepare_browser_video(original, preview)
                    original.unlink(missing_ok=True)
                st.session_state["original_preview"] = (identity, str(preview))
            except (OSError, RuntimeError, ValueError):
                st.info(
                    "A prévia não está disponível. Você ainda pode tentar analisar o vídeo."
                )
            finally:
                original.unlink(missing_ok=True)
        cached = st.session_state.get("original_preview")
        if kind == "video" and cached and cached[0] == identity:
            st.video(cached[1], format="video/mp4")


def history_page() -> None:
    st.subheader("Registros anteriores")
    point = st.text_input("Buscar pelo nome do local", placeholder="Ex.: garagem")
    try:
        response = api_request(
            "GET",
            "/vision/events",
            params={"limit": 30, **({"point_id": point} if point else {})},
        )
        response.raise_for_status()
        payload = response.json()
    except (RuntimeError, httpx.HTTPStatusError):
        st.info(
            "Não foi possível carregar os registros agora. Confira se iniciar.ps1 está em execução."
        )
        return
    if not payload["items"]:
        st.info("Seus registros aparecerão aqui depois da primeira análise.")
    for item in payload["items"]:
        with st.container(border=True):
            when = (
                datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                .astimezone()
                .strftime("%d/%m/%Y às %H:%M")
            )
            st.markdown(
                f"**{html.escape(item.get('point_id') or 'Local não informado')}** · {when}"
            )
            names = [LABELS.get(name, name) for name in item["detected_classes"]]
            st.write(
                "⚠️ Verificar: " + ", ".join(names)
                if names
                else "🔎 Nenhum sinal identificado pela análise"
            )
            st.caption("Vídeo" if item["input_type"] == "video" else "Foto")


def main() -> None:
    st.set_page_config(
        page_title="DrenaVision · Cuidar do local",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        "<style>.block-container{max-width:1120px;padding-top:2rem} [data-testid=stMetricValue]{font-size:1.5rem} [data-testid=stVideo]{max-height:580px} div[data-testid=stAlert]{border-radius:12px} </style>",
        unsafe_allow_html=True,
    )
    st.title("💧 DrenaVision")
    st.caption("Um olhar a mais para cuidar dos pontos de drenagem.")
    analyze, history, about = st.tabs(
        ["Analisar foto ou vídeo", "Registros anteriores", "Como funciona"]
    )
    with analyze:
        analysis_page()
    with history:
        history_page()
    with about:
        st.subheader("Do registro à verificação")
        st.write(
            "1. Escolha uma foto, um vídeo ou um exemplo pronto.\n\n2. Toque em **Analisar foto ou vídeo**.\n\n3. Confira os contornos coloridos e os avisos. Quando houver sinais, peça uma verificação à equipe responsável."
        )
        st.info(
            "A ferramenta pode errar e deixar passar objetos. As marcações indicam o que a análise encontrou; uma pessoa confirma a situação no local."
        )
        st.write(
            "O protótipo procura resíduos ou possíveis obstruções, água acumulada e animais visíveis. Ele não mede profundidade nem prevê alagamentos."
        )
        st.caption(
            "Os exemplos mantêm seus créditos e licenças. Filmagens e fotos ilustrativas são analisadas pelo mesmo modelo usado nos seus arquivos."
        )


if __name__ == "__main__":
    main()
