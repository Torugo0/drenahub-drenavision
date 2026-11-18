"""Montagem local de capturas reais da interface, com tempo original e voz pt-BR.

Não captura/controla o navegador e não gera respostas da aplicação. A entrada é
um JSONL de JPEGs reais ``{index, path, t_ms}``, na ordem da captura, e capítulos
JSON com ``id, title, narration, start_ms, end_ms``. Blocos realmente capturados
podem vir de ``--segments-index`` (JSONL com start_ms/end_ms) ou ``segments``
por capítulo. Somente o tempo sem captura entre blocos é removido; o tempo
interno, inclusive espera e reprodução, permanece em velocidade 1x. A inferência
de blocos por lacunas >1000ms só é autorizada para intro/foto_bueiro, os dois
capítulos anteriores ao registro explícito. Cards explícitos usam
``type: card`` e ``lines``. Qualquer pausa adicional em uma captura exige
``result_ms`` ou ``allow_hold: true`` e recebe identificação na imagem.

Exemplo (PowerShell, na raiz):
  .\\.venv\\Scripts\\python.exe ferramentas/scripts/montar_demonstracao.py `
    --frames-index resultados/local/gravacao/captura/frames.jsonl `
    --segments-index resultados/local/gravacao/captura/segments.jsonl `
    --chapters resultados/local/gravacao/capitulos.json `
    --output resultados/local/gravacao/drenavision_demonstracao_completa.mp4 `
    --work-dir resultados/local/gravacao/montagem

O MP4 usa 24 FPS como formato de distribuição. O manifesto registra a cadência
real e os intervalos da captura: duplicar um frame não cria movimento novo.
Originais e intermediários são preservados. Não há publicação externa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class CapturedFrame:
    index: int
    path: Path
    t_ms: float


def load_frames(index_path: Path) -> list[CapturedFrame]:
    frames: list[CapturedFrame] = []
    for line_number, line in enumerate(
        index_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        frame = CapturedFrame(
            index=int(row["index"]),
            path=(index_path.parent / row["path"]).resolve(),
            t_ms=float(row["t_ms"]),
        )
        if not math.isfinite(frame.t_ms) or frame.t_ms < 0:
            raise ValueError(f"Timestamp inválido na linha {line_number}")
        if frames and (
            frame.index <= frames[-1].index or frame.t_ms <= frames[-1].t_ms
        ):
            raise ValueError("Frames devem ter índice e timestamp crescentes")
        if not frame.path.is_file():
            raise FileNotFoundError(frame.path)
        frames.append(frame)
    if not frames:
        raise ValueError("O índice não contém frames")
    return frames


def select_timeline(
    frames: list[CapturedFrame], start_ms: float, end_ms: float
) -> list[tuple[CapturedFrame, float]]:
    """Seleção inclusiva, com durações que somam o intervalo real solicitado.

    O primeiro frame cobre também os poucos milissegundos entre o marcador e
    a próxima captura. Lacunas internas mantêm sua duração, sem aceleração.
    """
    if not all(math.isfinite(t) for t in (start_ms, end_ms)):
        raise ValueError("Os limites precisam ser finitos")
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("Intervalo de capítulo inválido")
    selected = [frame for frame in frames if start_ms <= frame.t_ms <= end_ms]
    if not selected:
        raise ValueError("Nenhum frame no intervalo do capítulo")
    intervals = [b.t_ms - a.t_ms for a, b in zip(frames, frames[1:], strict=False)]
    tolerance = max(1000.0, statistics.median(intervals) * 3) if intervals else 1000
    if selected[0].t_ms - start_ms > tolerance:
        raise ValueError("Faltam frames no início do capítulo")
    if end_ms - selected[-1].t_ms > tolerance:
        raise ValueError("O fim do capítulo ultrapassa a captura disponível")
    timeline = []
    for position, frame in enumerate(selected):
        begins = start_ms if position == 0 else frame.t_ms
        ends = selected[position + 1].t_ms if position + 1 < len(selected) else end_ms
        timeline.append((frame, max(0, ends - begins) / 1000))
    if not timeline:
        raise ValueError("O capítulo não tem duração reproduzível")
    return timeline


def validate_chapters(chapters: list[dict]) -> None:
    seen = set()
    last_capture_end = -1.0
    for chapter in chapters:
        chapter_id = chapter["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", chapter_id) or chapter_id in seen:
            raise ValueError("Cada capítulo precisa de id único (letras/números/_/-)")
        seen.add(chapter_id)
        if chapter.get("type", "capture") == "card":
            continue
        start, end = float(chapter["start_ms"]), float(chapter["end_ms"])
        if start < last_capture_end:
            raise ValueError("Capítulos de captura estão fora de ordem ou sobrepostos")
        if end <= start:
            raise ValueError("O fim precisa vir depois do início do capítulo")
        last_capture_end = end


def validate_segments(segments: list[dict]) -> list[dict]:
    checked = []
    previous_end = -1.0
    for segment in segments:
        start, end = float(segment["start_ms"]), float(segment["end_ms"])
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise ValueError("Intervalo de bloco de captura inválido")
        if start < previous_end:
            raise ValueError("Blocos de captura estão fora de ordem ou sobrepostos")
        checked.append({"start_ms": start, "end_ms": end})
        previous_end = end
    return checked


def load_segments(index_path: Path) -> list[dict]:
    return validate_segments(
        [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    )


def infer_recorded_blocks(
    frames: list[CapturedFrame],
    start_ms: float,
    end_ms: float,
    gap_ms: float = 1000,
) -> list[dict]:
    """Compatibilidade estrita com os dois capítulos cuja captura foi auditada.

    O último frame de um bloco cobre um intervalo nominal de amostragem; o
    restante da lacuna não gravada é cortado. A inferência é declarada no
    manifesto e nunca aplicada automaticamente aos capítulos posteriores.
    """
    selected = [frame for frame in frames if start_ms <= frame.t_ms <= end_ms]
    if not selected:
        raise ValueError("Nenhum frame para inferir os blocos do capítulo")
    regular_intervals = [
        b.t_ms - a.t_ms
        for a, b in zip(selected, selected[1:], strict=False)
        if b.t_ms - a.t_ms <= gap_ms
    ]
    nominal = statistics.median(regular_intervals) if regular_intervals else 125.0
    groups: list[list[CapturedFrame]] = [[selected[0]]]
    for frame in selected[1:]:
        if frame.t_ms - groups[-1][-1].t_ms > gap_ms:
            groups.append([])
        groups[-1].append(frame)
    blocks = []
    for position, group in enumerate(groups):
        begins = group[0].t_ms
        if position == 0 and begins - start_ms < gap_ms:
            begins = start_ms
        ends = min(end_ms, group[-1].t_ms + nominal)
        if position == len(groups) - 1 and end_ms - group[-1].t_ms < gap_ms:
            ends = end_ms
        blocks.append({"start_ms": begins, "end_ms": ends})
    return validate_segments(blocks)


def ensure_processing_is_uncut(chapter: dict, blocks: list[dict]) -> None:
    if chapter.get("process_start_ms") is None or chapter.get("result_ms") is None:
        return
    process_start, result = (
        float(chapter["process_start_ms"]),
        float(chapter["result_ms"]),
    )
    if result < process_start:
        raise ValueError("O resultado não pode anteceder o início do processamento")
    covered_until = process_start
    for block in blocks:
        if block["end_ms"] < covered_until:
            continue
        if block["start_ms"] > covered_until + 1.0:
            break
        covered_until = max(covered_until, block["end_ms"])
        if covered_until >= result:
            return
    raise ValueError(
        f"{chapter['id']}: um corte removeria parte da espera do processamento"
    )


def chapter_timeline(
    frames: list[CapturedFrame],
    chapter: dict,
    global_segments: list[dict] | None,
) -> tuple[list[tuple[CapturedFrame, float]], list[dict], str]:
    start, end = float(chapter["start_ms"]), float(chapter["end_ms"])
    candidates = chapter.get("segments")
    origin = "chapter_explicit"
    if candidates is None and global_segments is not None:
        candidates = global_segments
        origin = "global_explicit"
    blocks = []
    if candidates is not None:
        for block in validate_segments(candidates):
            begins, ends = max(start, block["start_ms"]), min(end, block["end_ms"])
            if ends > begins:
                blocks.append({"start_ms": begins, "end_ms": ends})
    if not blocks:
        if chapter["id"] in {"intro", "foto_bueiro"}:
            blocks = infer_recorded_blocks(frames, start, end)
            origin = "audited_legacy_inference_gap_over_1000ms"
        elif candidates is None:
            blocks = [{"start_ms": start, "end_ms": end}]
            origin = "continuous_chapter_range"
        else:
            raise ValueError(
                f"{chapter['id']}: nenhum bloco explícito intersecta o capítulo"
            )
    ensure_processing_is_uncut(chapter, blocks)
    timeline = []
    for block in blocks:
        timeline.extend(select_timeline(frames, block["start_ms"], block["end_ms"]))
    return timeline, blocks, origin


def ffconcat_quote(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "'\\''") + "'"


def write_concat(path: Path, timeline: list[tuple[CapturedFrame, float]]) -> None:
    lines = ["ffconcat version 1.0"]
    for frame, seconds in timeline:
        lines.extend([f"file {ffconcat_quote(frame.path)}", f"duration {seconds:.9f}"])
    # O demuxer só aplica a duração do último frame se outro arquivo o seguir.
    lines.append(f"file {ffconcat_quote(timeline[-1][0].path)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def caption_parts(text: str, limit: int = 104) -> list[str]:
    chunks = []
    for sentence in re.split(r"(?<=[.!?;])\s+", " ".join(text.split())):
        words, current = sentence.split(), ""
        for word in words:
            if current and len(current) + len(word) + 1 > limit:
                chunks.append(current)
                current = ""
            current = f"{current} {word}".strip()
        if current:
            chunks.append(current)
    return chunks


def wave_seconds(path: Path) -> float:
    with wave.open(str(path)) as stream:
        return stream.getnframes() / stream.getframerate()


def synthesize_narration(
    text: str, directory: Path, rate: int
) -> tuple[Path, list[dict]]:
    """SAPI local; texto e nomes de arquivo passam por JSON, nunca por shell."""
    parts = caption_parts(text)
    tasks = [
        {"text": part, "path": str(directory / f"fala_{index:03d}.wav")}
        for index, part in enumerate(parts)
    ]
    (directory / "falas.json").write_text(
        json.dumps(tasks, ensure_ascii=False), encoding="utf-8"
    )
    script = directory / "narrar.ps1"
    script.write_text(
        """param([string]$TasksPath, [int]$VoiceRate = 0)
$ErrorActionPreference = 'Stop'
$taskVoice = New-Object -ComObject SAPI.SpVoice
$taskToken = @($taskVoice.GetVoices() | Where-Object {
    $_.GetAttribute('Language') -eq '416'
})[0]
if ($null -eq $taskToken) { throw 'Voz SAPI pt-BR não instalada' }
$taskVoice.Voice = $taskToken
$taskVoice.Rate = $VoiceRate
$taskItems = Get-Content -LiteralPath $TasksPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($taskItem in $taskItems) {
    $taskStream = New-Object -ComObject SAPI.SpFileStream
    try {
        $taskStream.Open($taskItem.path, 3, $false)
        $taskVoice.AudioOutputStream = $taskStream
        [void]$taskVoice.Speak($taskItem.text)
    } finally { $taskStream.Close() }
}
""",
        encoding="utf-8-sig",
    )
    if tasks:
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-File",
                str(script),
                "-TasksPath",
                str(directory / "falas.json"),
                "-VoiceRate",
                str(rate),
            ],
            check=True,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    narration = directory / "narracao.wav"
    captions, position = [], 0.35
    with wave.open(str(narration), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(22050)
        output.writeframes(b"\x00" * round(position * 22050) * 2)
        for task in tasks:
            with wave.open(task["path"], "rb") as source:
                if (
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getframerate(),
                ) != (1, 2, 22050):
                    raise ValueError(
                        "Formato SAPI inesperado; esperado PCM mono 22050Hz/16bit"
                    )
                duration = source.getnframes() / source.getframerate()
                output.writeframes(source.readframes(source.getnframes()))
            captions.append(
                {"start": position, "end": position + duration, "text": task["text"]}
            )
            output.writeframes(b"\x00" * round(0.16 * 22050) * 2)
            position += duration + 0.16
    return narration, captions


def silence(path: Path, seconds: float = 0.1) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(22050)
        stream.writeframes(b"\x00" * round(seconds * 22050) * 2)


def write_narration_timeline(
    path: Path,
    narration_paths: list[Path],
    durations: list[float],
) -> None:
    """Um único PCM evita acumular padding de um codificador AAC por capítulo."""
    if len(narration_paths) != len(durations):
        raise ValueError("Áudios e durações precisam corresponder")
    position, elapsed = 0, 0.0
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(22050)
        for audio, duration in zip(narration_paths, durations, strict=True):
            elapsed += duration
            target_end = round(elapsed * 22050)
            allocated = target_end - position
            with wave.open(str(audio), "rb") as source:
                if (
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getframerate(),
                ) != (1, 2, 22050):
                    raise ValueError("A montagem requer PCM mono 22050Hz/16bit")
                frames = source.getnframes()
                if frames > allocated:
                    raise ValueError("A narração ultrapassa a duração do capítulo")
                output.writeframes(source.readframes(frames))
            output.writeframes(b"\x00" * (allocated - frames) * 2)
            position = target_end


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    return f"{centiseconds // 360000}:{centiseconds // 6000 % 60:02}:{centiseconds // 100 % 60:02}.{centiseconds % 100:02}"


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    return f"{milliseconds // 3600000:02}:{milliseconds // 60000 % 60:02}:{milliseconds // 1000 % 60:02},{milliseconds % 1000:03}"


def escape_ass(text: str) -> str:
    return (
        text.replace("\\", "＼")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", r"\N")
    )


def write_ass(
    path: Path,
    title: str,
    captions: list[dict],
    width: int,
    height: int,
    duration: float,
    source_duration: float,
    is_card: bool,
) -> None:
    font = round(height * 0.031)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Segoe UI,{font},&H00FFFFFF,&H00FFFFFF,&H000B1729,&H000B1729,0,0,0,0,100,100,0,0,1,1,0,2,45,45,14,1
Style: Title,Segoe UI,{round(height * 0.029)},&H00FFFFFF,&H00FFFFFF,&H000B1729,&H000B1729,-1,0,0,0,100,100,0,0,1,0,0,7,26,26,10,1
Style: Note,Segoe UI,{round(height * 0.019)},&H00ADD8FA,&H00ADD8FA,&H000B1729,&H000B1729,0,0,0,0,100,100,0,0,1,0,0,9,20,20,14,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = [
        f"Dialogue: 0,0:00:00.00,{ass_time(min(3, duration))},Title,,0,0,0,,{escape_ass(title)}"
    ]
    label = (
        "Cartão explicativo" if is_card else "Captura real por quadros • tempo original"
    )
    events.append(
        f"Dialogue: 0,0:00:00.00,{ass_time(min(source_duration, duration))},Note,,0,0,0,,{label}"
    )
    if not is_card and duration > source_duration + 0.04:
        events.append(
            f"Dialogue: 0,{ass_time(source_duration)},{ass_time(duration)},Note,,0,0,0,,Pausa de explicação • resultado já concluído"
        )
    for caption in captions:
        events.append(
            f"Dialogue: 0,{ass_time(caption['start'])},{ass_time(caption['end'])},Caption,,0,0,0,,{escape_ass(caption['text'])}"
        )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")


def wrap_draw(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def make_card(chapter: dict, path: Path, size: tuple[int, int]) -> None:
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        fraction = y / max(height - 1, 1)
        draw.line(
            (0, y, width, y),
            fill=(13, round(38 + fraction * 20), round(73 + fraction * 30)),
        )
    font_root = Path("C:/Windows/Fonts")
    title_font = ImageFont.truetype(
        str(font_root / "segoeuib.ttf"), round(height * 0.067)
    )
    body_font = ImageFont.truetype(
        str(font_root / "segoeui.ttf"), round(height * 0.044)
    )
    small_font = ImageFont.truetype(
        str(font_root / "segoeui.ttf"), round(height * 0.027)
    )
    margin, y = round(width * 0.06), round(height * 0.09)
    draw.text(
        (margin, y), "DRENAVISION  |  DEMONSTRAÇÃO", font=small_font, fill="#91D9FF"
    )
    y += round(height * 0.09)
    for line in wrap_draw(chapter["title"], title_font, width - margin * 2):
        draw.text((margin, y), line, font=title_font, fill="white")
        y += round(height * 0.083)
    y += round(height * 0.035)
    for paragraph in chapter.get("lines", []):
        for line in wrap_draw(str(paragraph), body_font, width - margin * 2):
            if y > height * 0.88:
                raise ValueError(f"Texto demais no card {chapter['id']}")
            draw.text((margin, y), line, font=body_font, fill="#DFEDFB")
            y += round(height * 0.059)
        y += round(height * 0.025)
    image.save(path)


def run_ffmpeg(arguments: list[str], log_path: Path, cwd: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-y", *arguments],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if process.returncode:
        raise RuntimeError(f"FFmpeg falhou. Consulte {log_path}")


def source_cadence(frames: list[CapturedFrame]) -> dict:
    intervals = [
        (b.t_ms - a.t_ms) / 1000 for a, b in zip(frames, frames[1:], strict=False)
    ]
    return {
        "frames": len(frames),
        "span_seconds": (frames[-1].t_ms - frames[0].t_ms) / 1000,
        "average_fps": (len(frames) - 1) / sum(intervals) if intervals else 0,
        "median_interval_seconds": statistics.median(intervals) if intervals else 0,
        "max_interval_seconds": max(intervals, default=0),
        "intervals_over_1_second": sum(interval > 1 for interval in intervals),
    }


def recorded_block_cadence(frames: list[CapturedFrame], blocks: list[dict]) -> dict:
    intervals = []
    count = 0
    for block in blocks:
        selected = [
            frame
            for frame in frames
            if block["start_ms"] <= frame.t_ms <= block["end_ms"]
        ]
        count += len(selected)
        intervals.extend(
            (b.t_ms - a.t_ms) / 1000
            for a, b in zip(selected, selected[1:], strict=False)
        )
    return {
        "sampled_frames": count,
        "average_fps_excluding_between_block_gaps": len(intervals) / sum(intervals)
        if intervals
        else 0,
        "median_interval_seconds": statistics.median(intervals) if intervals else 0,
        "max_interval_seconds_inside_block": max(intervals, default=0),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(
    frames_index: Path,
    chapters_path: Path,
    output: Path,
    work_dir: Path,
    size: tuple[int, int] = (1280, 720),
    fps: int = 24,
    rate: int = 0,
    no_narration: bool = False,
    segments_index: Path | None = None,
) -> dict:
    frames_index, chapters_path = frames_index.resolve(), chapters_path.resolve()
    output, work_dir = output.resolve(), work_dir.resolve()
    if output.exists():
        raise FileExistsError(f"A saída já existe; escolha outro nome: {output}")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(
            f"O diretório de montagem precisa estar vazio: {work_dir}"
        )
    frames = load_frames(frames_index)
    global_segments = (
        load_segments(segments_index.resolve()) if segments_index else None
    )
    payload = json.loads(chapters_path.read_text(encoding="utf-8-sig"))
    chapters = payload if isinstance(payload, list) else payload["chapters"]
    if not chapters:
        raise ValueError("Nenhum capítulo foi informado")
    validate_chapters(chapters)
    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    header_height, subtitle_height = round(height * 0.056), round(height * 0.111)
    content_height = height - header_height - subtitle_height
    with Image.open(frames[0].path) as first_frame_image:
        source_size = list(first_frame_image.size)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "real_browser_screenshots_in_recorded_blocks",
        "source_index_sha256": sha256(frames_index),
        "chapters_sha256": sha256(chapters_path),
        "capture_cadence": source_cadence(frames),
        "capture_cadence_includes_uncaptured_orchestration_gaps": True,
        "source_first_frame_size": source_size,
        "source_playback_speed": 1.0,
        "cut_policy": "only uncaptured orchestration intervals between recorded blocks; no processing/playback speedup",
        "segments_index_sha256": sha256(segments_index) if segments_index else None,
        "output_fps": fps,
        "output_fps_is_distribution_format": True,
        "output_size": list(size),
        "narration": "none"
        if no_narration
        else "offline Windows SAPI Microsoft Maria pt-BR (synthetic)",
        "narration_rate": rate,
        "ui_is_generated": False,
        "chapters": [],
    }
    all_captions, segments, narration_paths, offset = [], [], [], 0.0
    for number, chapter in enumerate(chapters, 1):
        directory = work_dir / f"{number:02d}_{chapter['id']}"
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[{number}/{len(chapters)}] {chapter['title']}", flush=True)
        if no_narration or not chapter.get("narration", "").strip():
            audio, captions = directory / "narracao.wav", []
            silence(audio)
        else:
            audio, captions = synthesize_narration(
                chapter["narration"], directory, rate
            )
        is_card = chapter.get("type", "capture") == "card"
        if is_card:
            source_duration = max(
                float(chapter.get("min_duration", 4)), wave_seconds(audio) + 0.7
            )
            card = directory / "card.png"
            make_card(chapter, card, (width, content_height))
            inputs = ["-loop", "1", "-framerate", str(fps), "-i", str(card)]
            selected = []
            source_blocks, blocks_origin = [], "explanatory_card"
        else:
            start, end = float(chapter["start_ms"]), float(chapter["end_ms"])
            selected, source_blocks, blocks_origin = chapter_timeline(
                frames, chapter, global_segments
            )
            source_duration = sum(seconds for _, seconds in selected)
            concat = directory / "frames.ffconcat"
            write_concat(concat, selected)
            inputs = ["-f", "concat", "-safe", "0", "-i", str(concat)]
        duration = (
            math.ceil(
                max(
                    source_duration,
                    float(chapter.get("min_duration", 0)),
                    wave_seconds(audio) + 0.7,
                )
                * fps
            )
            / fps
        )
        hold = duration - source_duration
        if not is_card and hold > 1 / fps + 0.001:
            result_ms = chapter.get("result_ms")
            confirmed = chapter.get("allow_hold") is True or (
                result_ms is not None
                and float(chapter["start_ms"])
                <= float(result_ms)
                <= float(chapter["end_ms"])
            )
            if not confirmed:
                raise ValueError(
                    f"{chapter['id']}: a narração excede a captura em {hold:.2f}s. Informe result_ms/allow_hold só após um resultado, ou encurte a fala."
                )
        subtitle = directory / "legendas.ass"
        write_ass(
            subtitle,
            chapter["title"],
            captions,
            width,
            height,
            duration,
            source_duration,
            is_card,
        )
        video_filter = (
            f"fps={fps},scale={width}:{content_height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad={width}:{height}:(ow-iw)/2:{header_height}:color=0x0B1729,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={max(0, hold):.6f},"
            "subtitles=legendas.ass"
        )
        segment = directory / "segmento.mp4"
        run_ffmpeg(
            [
                *inputs,
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                video_filter,
                "-af",
                "apad",
                "-t",
                f"{duration:.9f}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(segment),
            ],
            directory / "ffmpeg.log",
            directory,
        )
        for caption in captions:
            all_captions.append(
                {
                    **caption,
                    "start": caption["start"] + offset,
                    "end": caption["end"] + offset,
                }
            )
        manifest["chapters"].append(
            {
                "id": chapter["id"],
                "title": chapter["title"],
                "type": "card" if is_card else "capture",
                "output_start_seconds": offset,
                "output_duration_seconds": duration,
                "source_start_ms": chapter.get("start_ms"),
                "source_end_ms": chapter.get("end_ms"),
                "source_duration_seconds": source_duration,
                "source_frames_selected": len(selected),
                "source_recorded_blocks": source_blocks,
                "recorded_block_cadence": None
                if is_card
                else recorded_block_cadence(frames, source_blocks),
                "blocks_origin": blocks_origin,
                "uncaptured_orchestration_seconds_removed": 0
                if is_card
                else max(0, (end - start) / 1000 - source_duration),
                "explanation_hold_seconds": 0 if is_card else max(0, hold),
                "narration_seconds": wave_seconds(audio),
                "narration_text": chapter.get("narration", ""),
                "segment_sha256": sha256(segment),
            }
        )
        segments.append(segment)
        narration_paths.append(audio)
        offset += duration
    final_concat = work_dir / "segmentos.ffconcat"
    final_concat.write_text(
        "ffconcat version 1.0\n"
        + "\n".join(f"file {ffconcat_quote(path)}" for path in segments)
        + "\n",
        encoding="utf-8",
    )
    full_narration = work_dir / "narracao_completa.wav"
    write_narration_timeline(
        full_narration,
        narration_paths,
        [chapter["output_duration_seconds"] for chapter in manifest["chapters"]],
    )
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(final_concat),
            "-i",
            str(full_narration),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            f"{offset:.9f}",
            "-movflags",
            "+faststart",
            str(output),
        ],
        work_dir / "ffmpeg_final.log",
        work_dir,
    )
    srt = output.with_suffix(".srt")
    srt.write_text(
        "\n\n".join(
            f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}"
            for i, c in enumerate(all_captions, 1)
        )
        + "\n",
        encoding="utf-8",
    )
    transcript = output.with_suffix(".txt")
    transcript.write_text(
        "Narração sintética local: Microsoft Maria, português do Brasil.\n"
        "Captura real por quadros; intervalos de orquestração sem captura foram cortados.\n"
        "Tempo dentro dos blocos, inclusive processamento e reprodução, preservado em velocidade 1x.\n\n"
        + "\n\n".join(f"{c['title']}\n{c.get('narration', '')}" for c in chapters),
        encoding="utf-8",
    )
    manifest.update(
        {
            "duration_seconds": offset,
            "output_bytes": output.stat().st_size,
            "output_sha256": sha256(output),
            "audio_assembly": "single AAC encode from narration PCM aligned to the video chapter timeline; no cumulative per-segment AAC padding",
        }
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output), "seconds": offset, "bytes": output.stat().st_size},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frames-index", type=Path, required=True)
    parser.add_argument("--chapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--segments-index", type=Path)
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--voice-rate", type=int, default=0, choices=range(-3, 4))
    parser.add_argument("--no-narration", action="store_true")
    args = parser.parse_args()
    try:
        width, height = map(int, args.size.lower().split("x"))
        if (
            width < 640
            or height < 360
            or width % 2
            or height % 2
            or not 1 <= args.fps <= 60
        ):
            raise ValueError("Use resolução par >=640x360 e FPS entre 1 e 60")
        assemble(
            args.frames_index,
            args.chapters,
            args.output,
            args.work_dir,
            (width, height),
            args.fps,
            args.voice_rate,
            args.no_narration,
            args.segments_index,
        )
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
