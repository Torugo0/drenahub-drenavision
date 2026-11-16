"""Render YOLO labels over source images for human inspection."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .constants import CLASS_NAMES, IMAGE_SUFFIXES, SPLITS
from .yolo import YoloBox, read_yolo

COLORS = {
    0: (239, 83, 80),
    1: (33, 150, 243),
    2: (255, 193, 7),
}


def _rectangle(box: YoloBox, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = round((box.x_center - box.width / 2) * width)
    y1 = round((box.y_center - box.height / 2) * height)
    x2 = round((box.x_center + box.width / 2) * width)
    y2 = round((box.y_center + box.height / 2) * height)
    return x1, y1, x2, y2


def render_annotation(image_path: Path, label_path: Path, output_path: Path) -> Path:
    """Draw one strict YOLO annotation file and save a JPEG/PNG preview."""

    boxes = read_yolo(label_path, class_count=len(CLASS_NAMES))
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 250))
    for box in boxes:
        color = COLORS[box.class_id]
        rectangle = _rectangle(box, *image.size)
        draw.rectangle(rectangle, outline=color, width=line_width)
        label = f"{box.class_id}: {CLASS_NAMES[box.class_id]}"
        text_bbox = draw.textbbox((rectangle[0], rectangle[1]), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_top = max(0, rectangle[1] - text_height - 4)
        draw.rectangle(
            (
                rectangle[0],
                label_top,
                rectangle[0] + text_width + 6,
                label_top + text_height + 4,
            ),
            fill=color,
        )
        draw.text((rectangle[0] + 3, label_top + 2), label, fill=(0, 0, 0), font=font)
    if not boxes:
        draw.rectangle((0, 0, 185, 22), fill=(40, 40, 40))
        draw.text((5, 5), "negative sample (no boxes)", fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return output_path


def visualize_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    samples_per_split: int = 6,
    seed: int = 42,
) -> list[Path]:
    """Render a deterministic random sample from every dataset split."""

    if samples_per_split < 0:
        raise ValueError("samples_per_split must be non-negative")
    rendered: list[Path] = []
    for split_index, split in enumerate(SPLITS):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        candidates = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(candidates) < samples_per_split:
            raise RuntimeError(
                f"{split} contains {len(candidates)} images; cannot sample {samples_per_split}"
            )
        chosen = random.Random(seed + split_index).sample(candidates, samples_per_split)
        for image_path in sorted(chosen):
            label_path = label_dir / f"{image_path.stem}.txt"
            output_path = output_root / split / f"{image_path.stem}.jpg"
            rendered.append(render_annotation(image_path, label_path, output_path))
    return rendered


def create_contact_sheet(
    rendered_paths: list[Path],
    output_path: Path,
    *,
    columns: int = 3,
    cell_size: tuple[int, int] = (420, 320),
) -> Path:
    """Assemble annotated previews into a compact review sheet."""

    if not rendered_paths:
        raise ValueError("at least one rendered preview is required")
    if columns <= 0:
        raise ValueError("columns must be positive")
    label_height = 24
    rows = (len(rendered_paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * cell_size[0], rows * (cell_size[1] + label_height)),
        (24, 27, 33),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, path in enumerate(rendered_paths):
        column = index % columns
        row = index // columns
        x = column * cell_size[0]
        y = row * (cell_size[1] + label_height)
        with Image.open(path) as opened:
            preview = ImageOps.contain(
                opened.convert("RGB"), cell_size, Image.Resampling.LANCZOS
            )
        offset_x = x + (cell_size[0] - preview.width) // 2
        offset_y = y + (cell_size[1] - preview.height) // 2
        sheet.paste(preview, (offset_x, offset_y))
        label = f"{path.parent.name}/{path.name}"
        draw.text((x + 6, y + cell_size[1] + 6), label, fill=(235, 238, 245), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90, optimize=True)
    return output_path
