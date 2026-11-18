"""Render the audited Markdown report without running models or changing metrics.

Optional authoring dependencies: pip install -e ".[report]"
Run from the repository root: python ferramentas/scripts/gerar_relatorio.py --render
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/relatorio/relatorio_fonte.md"
OUTPUT = SOURCE.with_name("DrenaVision_Applied_Computer_Vision_FIAP.pdf")
NAVY = colors.HexColor("#15324F")
BLUE = colors.HexColor("#146A9C")
INK = colors.HexColor("#243647")
MUTED = colors.HexColor("#546578")
PALE = colors.HexColor("#EEF4F9")
RULE = colors.HexColor("#D7E1EA")
WIDTH, HEIGHT = A4
MARGIN = 21 * mm
CONTENT_WIDTH = WIDTH - 2 * MARGIN


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: str | Path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def section(source: str, heading: str, until: str) -> str:
    return source.split(heading, 1)[1].split(until, 1)[0].strip()


def audit_source(source: str, source_path: Path) -> dict:
    abstract = section(source, "## 1. Resumo", "**Palavras-chave:**")
    objective = section(source, "## 3. Objetivo", "## 4. Ferramentas")
    counts = {"resumo": len(abstract.split()), "objetivo": len(objective.split())}
    if counts != {"resumo": 250, "objetivo": 50}:
        raise ValueError(f"Wrong mandatory word counts: {counts}")
    for number in range(1, 12):
        if not re.search(rf"^## {number}\. ", source, re.MULTILINE):
            raise ValueError(f"Required section {number} absent")
    for name in (
        "Gabriel Machado Carrara Pimentel",
        "Lourenzo Ramos",
        "Vítor Hugo Rodrigues",
        "RM99880",
        "RM99951",
        "RM97758",
    ):
        if name not in source:
            raise ValueError(f"Missing team identity: {name}")
    training = read_json("resultados/metrics/v3/training_manifest.json")
    raw_metrics = read_json("resultados/metrics/v3/master_map_comparison.json")
    holdout = read_json("resultados/metrics/v4/evaluation_holdout40.json")
    regression = read_json("resultados/metrics/v4/evaluation_regression179.json")
    product = read_json("resultados/metrics/v4/product_validation.json")
    active = read_json("resultados/metrics/v4/active_profile.json")
    verified = []
    metric_section = section(source, "### 6.1 ", "### 6.2 ")
    for label, key in (
        ("Resíduos", "obstrucao_visivel"),
        ("Água", "agua_acumulada"),
        ("Animal", "animal_visivel"),
    ):
        values = raw_metrics["candidate"]["per_class"][key]
        cells = [f"{values[k]:.4f}".replace(".", ",") for k in (
            "precision", "recall", "map50", "map50_95"
        )]
        expected = "| " + " | ".join([label, *cells]) + " |"
        if expected not in metric_section:
            raise ValueError(f"YOLO metrics differ from artifact: {expected}")
        verified.append(expected)
    expected_epochs = (training["completed_epochs"], training["dataset"]["images_by_split"])
    if expected_epochs != (63, {"test": 60, "train": 1107, "val": 60}):
        raise ValueError("Training artifacts changed; review report text")
    if training["best_sha256"] not in source:
        raise ValueError("Report checkpoint differs from training evidence")
    for field in ("frozen_candidate", "configuration"):
        if digest(ROOT / active[field]) != active[f"{field}_sha256"]:
            raise ValueError(f"Frozen artifact changed: {field}")
    co = holdout["cohorts"]["holdout40"]
    for key in ("precision", "recall", "f1"):
        for mode in ("baseline", "candidate"):
            value = co[mode]["source_bbox_iou_0_5"]["obstrucao_visivel"][key]
            if f"{value:.4f}".replace(".", ",") not in source:
                raise ValueError(f"Missing holdout metric: {mode}/{key}")
    for cohort in ("test60", "extra119"):
        for mode in ("baseline", "candidate"):
            record = regression["cohorts"][cohort][mode]
            f1 = record["source_bbox_iou_0_5"]["animal_visivel"]["f1"]
            if f"{f1:.4f}".replace(".", ",") not in source:
                raise ValueError(f"Missing animal box tradeoff: {cohort}/{mode}")
    frames = sum(e["total_frames"] for e in product["entries"] if e["kind"] == "video")
    if frames != 1388 or product["status"] != "completed":
        raise ValueError("Functional product evidence differs from report")
    assets = []
    for relative in re.findall(r"^!\[[^\]]*\]\(([^\n]+)\)$", source, re.MULTILINE):
        path = (source_path.parent / relative).resolve()
        path.relative_to(source_path.parent.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        assets.append({"path": relative, "sha256": digest(path)})
    manifest = read_json(source_path.parent / "assets/manifesto_figuras.json")
    for figure in manifest["figures"]:
        if digest(ROOT / figure["output"]) != figure["output_sha256"]:
            raise ValueError(f"Figure differs from audited provenance: {figure['output']}")
    return {
        "word_count_method": "Unicode whitespace-delimited tokens; headings/keywords excluded",
        "word_counts": counts,
        "mandatory_sections": 11,
        "verified_metric_rows": verified,
        "training_epochs": training["completed_epochs"],
        "training_checkpoint_sha256": training["best_sha256"],
        "frozen_candidate_sha256": active["frozen_candidate_sha256"],
        "image_profile": active["image_profile"],
        "video_profile": active["video_profile"],
        "functional_video_frames": frames,
        "assets": assets,
        "figure_provenance_manifest_sha256": digest(source_path.parent / "assets/manifesto_figuras.json"),
        "figure_provenance_records": len(manifest.get("figures", manifest.get("assets", []))),
        "no_new_model_execution": True,
    }


def register_fonts() -> None:
    windows = Path("C:/Windows/Fonts")
    fallback = Path("/usr/share/fonts/truetype/dejavu")
    mappings = {
        "Body": ("arial.ttf", "DejaVuSans.ttf"),
        "BodyBold": ("arialbd.ttf", "DejaVuSans-Bold.ttf"),
        "BodyItalic": ("ariali.ttf", "DejaVuSans-Oblique.ttf"),
        "BodyBoldItalic": ("arialbi.ttf", "DejaVuSans-BoldOblique.ttf"),
        "Mono": ("consola.ttf", "DejaVuSansMono.ttf"),
    }
    for name, (win, linux) in mappings.items():
        target = windows / win
        if not target.is_file():
            target = fallback / linux
        if not target.is_file():
            raise FileNotFoundError(f"A Unicode font is required: {win} or {linux}")
        pdfmetrics.registerFont(TTFont(name, str(target)))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBoldItalic"
    )


def inline(text: str) -> str:
    """Translate the limited source dialect, preserving balanced URL parentheses."""
    links = {}
    start = 0
    while match := re.search(r"\[([^\]]+)\]\(", text[start:]):
        begin = start + match.start()
        opening = start + match.end() - 1
        end = opening + 1
        depth = 1
        while end < len(text) and depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        if depth:
            break
        url = text[opening + 1 : end - 1]
        label = match.group(1)
        token = f"LINKTOKEN{len(links)}END"
        links[token] = f'<link href="{html.escape(url, quote=True)}" color="#146A9C">{html.escape(label)}</link>'
        text = text[:begin] + token + text[end:]
        start = begin + len(token)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Mono" size="8.2">\1</font>', text)
    for key, value in links.items():
        text = text.replace(key, value)
    return text


def styles():
    base = getSampleStyleSheet()
    config = {
        "BodyText": dict(fontName="Body", fontSize=10.1, leading=14.0, textColor=INK,
                         alignment=TA_JUSTIFY, spaceAfter=7, allowWidows=0, allowOrphans=0),
        "Heading1": dict(fontName="BodyBold", fontSize=27, leading=32, textColor=NAVY, spaceAfter=16),
        "Heading2": dict(fontName="BodyBold", fontSize=15, leading=20, textColor=NAVY, spaceBefore=5, spaceAfter=12),
        "Heading3": dict(fontName="BodyBold", fontSize=11.4, leading=15.5, textColor=NAVY, spaceBefore=11, spaceAfter=8),
        "Heading4": dict(fontName="BodyBold", fontSize=10.5, leading=14.2, textColor=BLUE, spaceBefore=9, spaceAfter=7),
    }
    for name, options in config.items():
        for key, value in options.items():
            setattr(base[name], key, value)
    base.add(ParagraphStyle("Caption", parent=base["BodyText"], fontName="BodyItalic", fontSize=8.4,
                            leading=11.4, alignment=TA_LEFT, textColor=MUTED, spaceAfter=10))
    base.add(ParagraphStyle("TableCell", fontName="Body", fontSize=8.5, leading=11.6, textColor=INK,
                            splitLongWords=True, spaceAfter=0))
    base.add(ParagraphStyle("TableHead", parent=base["TableCell"], fontName="BodyBold", fontSize=8.2,
                            textColor=colors.white))
    base.add(ParagraphStyle("ReportCode", fontName="Mono", fontSize=7.5, leading=10.2, leftIndent=9,
                            rightIndent=9, spaceBefore=8, spaceAfter=12, textColor=INK,
                            backColor=PALE, borderColor=RULE, borderWidth=0.5, borderPadding=9))
    base.add(ParagraphStyle("CoverText", parent=base["BodyText"], alignment=TA_LEFT,
                            fontSize=11, leading=16, textColor=NAVY, spaceAfter=8))
    base.add(ParagraphStyle("CoverSub", parent=base["Heading2"], fontSize=18, leading=24, spaceAfter=22))
    base.add(ParagraphStyle("Reference", parent=base["BodyText"], alignment=TA_LEFT,
                            fontSize=9.1, leading=12.8, spaceAfter=10))
    return base


class Architecture(Flowable):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.height = 246

    def draw(self):
        c = self.canv

        def box(x, y, width, height, title, subtitle="", strong=False):
            c.setFillColor(NAVY if strong else PALE)
            c.setStrokeColor(RULE)
            c.roundRect(x, y, width, height, 5, fill=1, stroke=not strong)
            c.setFillColor(colors.white if strong else NAVY)
            c.setFont("BodyBold", 9)
            c.drawCentredString(x + width / 2, y + height / 2 + (3 if subtitle else -3), title)
            if subtitle:
                c.setFont("Body", 7.4)
                c.drawCentredString(x + width / 2, y + height / 2 - 9, subtitle)

        def arrow(x1, y1, x2, y2):
            c.setStrokeColor(BLUE)
            c.setLineWidth(1)
            c.line(x1, y1, x2, y2)
            c.line(x2 - 3, y2 + 4, x2, y2)
            c.line(x2 + 3, y2 + 4, x2, y2)

        w = self.width
        box(0, 206, w, 36, "Foto ou vídeo na interface Streamlit", "Upload / exemplos / foto da câmera")
        arrow(w / 2, 205, w / 2, 192)
        box(0, 154, w, 36, "API FastAPI: validação e controle da inferência", "Modelos carregados uma vez; processamento local", True)
        part = (w - 16) / 3
        for x in (part / 2, w / 2, w - part / 2):
            arrow(x, 153, x, 138)
        box(0, 95, part, 41, "YOLO26n customizado", "Treinamento v3")
        box(part + 8, 95, part, 41, "DINO + SigLIP 2", "Propostas e verificação")
        box(2 * part + 16, 95, part, 41, "SegFormer B1", "Superfícies de água")
        for x in (part / 2, w / 2, w - part / 2):
            arrow(x, 94, x, 80)
        box(0, 42, w, 36, "Fusão de classes, caixas e regiões; JPEG / MP4 + JSON", "Vídeo: inferência em cada quadro e confirmação temporal de água")
        arrow(w / 2, 41, w / 2, 29)
        box(0, -7, w, 34, "Evento SQLite + evidências -> resposta HTTP -> interface", "Cartões, avisos em tela, reprodução, download e histórico")


class ReportDocument(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=22 * mm, bottomMargin=20 * mm, **kwargs)
        self.heading_pages = []
        frame = Frame(MARGIN, 20 * mm, CONTENT_WIDTH, HEIGHT - 42 * mm,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=self.decorate))

    def decorate(self, canvas, _doc):
        canvas.saveState()
        canvas.setTitle("DrenaVision - Applied Computer Vision - FIAP 2026.2")
        canvas.setAuthor("Gabriel Machado Carrara Pimentel; Lourenzo Ramos; Vítor Hugo Rodrigues")
        canvas.setSubject("Treinamento customizado, avaliação e integração do protótipo acadêmico DrenaVision 0.5")
        if self.page == 1:
            canvas.setFillColor(NAVY)
            canvas.rect(0, HEIGHT - 15 * mm, WIDTH, 15 * mm, fill=1, stroke=0)
            canvas.setFillColor(colors.white)
            canvas.setFont("BodyBold", 9)
            canvas.drawString(MARGIN, HEIGHT - 9 * mm, "FIAP  |  GLOBAL SOLUTION 2026.2  |  CITIZENTECH")
        else:
            canvas.setStrokeColor(RULE)
            canvas.line(MARGIN, HEIGHT - 15.8 * mm, WIDTH - MARGIN, HEIGHT - 15.8 * mm)
            canvas.setFillColor(MUTED)
            canvas.setFont("Body", 8)
            canvas.drawString(MARGIN, HEIGHT - 12 * mm, "DrenaVision | Applied Computer Vision")
            canvas.drawRightString(WIDTH - MARGIN, HEIGHT - 12 * mm, "Relatório técnico | 0.5.0")
        canvas.setStrokeColor(RULE)
        canvas.line(MARGIN, 15 * mm, WIDTH - MARGIN, 15 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Body", 7.5)
        canvas.drawString(MARGIN, 10 * mm, "DrenaHub | Protótipo acadêmico de inspeção assistida")
        canvas.drawRightString(WIDTH - MARGIN, 10 * mm, str(self.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name.startswith("Heading"):
            label = flowable.getPlainText()
            key = f"heading-{len(self.heading_pages)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(label, key, level=0, closed=False)
            self.heading_pages.append({"heading": label, "page": self.page})


def table_widths(rows):
    count = len(rows[0])
    if count == 2:
        ratios = [0.36, 0.64]
        if rows[0][0] == "Integrante":
            ratios = [0.80, 0.20]
    elif count == 3:
        ratios = [0.60, 0.20, 0.20]
        if rows[0][0] in {"Coorte", "ID"}:
            ratios = [0.20, 0.40, 0.40] if rows[0][0] == "Coorte" else [0.065, 0.535, 0.40]
    elif count == 4:
        ratios = [0.40, 0.20, 0.20, 0.20]
    elif count == 5:
        ratios = [0.32, 0.17, 0.17, 0.17, 0.17]
        if rows[0][0] == "Origem":
            ratios = [0.23, 0.105, 0.13, 0.105, 0.43]
    elif count == 6:
        ratios = [0.27, 0.10, 0.12, 0.13, 0.13, 0.25]
    else:
        ratios = [1 / count] * count
    return [CONTENT_WIDTH * value for value in ratios]


def parse_story(source: str, source_path: Path, style):
    story = [Spacer(1, 13 * mm)]
    lines = source.splitlines()
    index = 0
    cover = True
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            cover = False
            index += 1
            continue
        if line == "<!-- architecture -->":
            story.append(Architecture(CONTENT_WIDTH))
            story.append(Spacer(1, 9))
            index += 1
            continue
        if line.startswith("```"):
            content = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                content.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(content), style["ReportCode"]))
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in row):
                    rows.append(row)
                index += 1
            cells = [[Paragraph(inline(cell), style["TableHead" if r == 0 else "TableCell"])
                      for cell in row] for r, row in enumerate(rows)]
            table = Table(cells, colWidths=table_widths(rows), repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, BLUE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, RULE),
            ]))
            story.extend([table, Spacer(1, 10)])
            continue
        if match := re.fullmatch(r"!\[([^\]]*)\]\(([^\n]+)\)", line):
            asset = source_path.parent / match.group(2)
            picture = Image(str(asset))
            max_height = 206
            name = asset.name
            if "curva_pr" in name:
                max_height = 231
            elif "matriz_confusao" in name:
                max_height = 300
            elif "bueiro" in name:
                max_height = 218
            elif "roedor_grade" in name:
                max_height = 191
            elif "residuos_incompletos" in name:
                max_height = 244
            elif "erro_pvc" in name:
                max_height = 265
            elif "negativo_" in name:
                max_height = 190
            ratio = min(CONTENT_WIDTH / picture.imageWidth, max_height / picture.imageHeight)
            picture.drawWidth = picture.imageWidth * ratio
            picture.drawHeight = picture.imageHeight * ratio
            picture.hAlign = "CENTER"
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            group = [picture, Spacer(1, 6)]
            if index < len(lines) and lines[index].startswith("*Figura "):
                group.append(Paragraph(inline(lines[index].strip()[1:-1]), style["Caption"]))
                index += 1
            story.append(KeepTogether(group))
            continue
        if match := re.match(r"^(#{1,4}) (.+)$", line):
            depth = len(match.group(1))
            selected = f"Heading{depth}"
            if cover and depth == 2:
                selected = "CoverSub"
            story.append(Paragraph(inline(match.group(2)), style[selected]))
            index += 1
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip():
            paragraph.append(lines[index].strip())
            index += 1
        text = " ".join(paragraph)
        selected = "CoverText" if cover else "BodyText"
        if text.startswith("*Figura ") and text.endswith("*"):
            text = text[1:-1]
            selected = "Caption"
        if re.match(r"^\[\d+\] ", text):
            selected = "Reference"
        story.append(Paragraph(inline(text), style[selected]))
    return story


def render_pdf(pdf_path: Path, destination: Path):
    import pypdfium2 as pdfium
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    destination.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    previews = []
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=1.35)
        image = bitmap.to_pil()
        target = destination / f"page_{index + 1:02d}.png"
        image.save(target)
        thumb = image.copy()
        thumb.thumbnail((397, 562))
        previews.append((index + 1, thumb))
        bitmap.close()
        page.close()
    for start in range(0, len(previews), 6):
        sheet = PILImage.new("RGB", (3 * 417, 2 * 600), "#e3eaf1")
        draw = ImageDraw.Draw(sheet)
        for index, (number, thumb) in enumerate(previews[start : start + 6]):
            x = (index % 3) * 417 + 10
            y = (index // 3) * 600 + 28
            draw.text((x, y - 24), f"Página {number}", font=font, fill="#15324f")
            sheet.paste(thumb, (x, y))
        sheet.save(destination / f"contact_{start // 6 + 1:02d}.jpg", quality=95)
    pdf.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    source = args.source.read_text(encoding="utf-8-sig")
    checks = audit_source(source, args.source)
    register_fonts()
    doc = ReportDocument(str(args.output))
    doc.build(parse_story(source, args.source, styles()))
    from pypdf import PdfReader

    reader = PdfReader(args.output)
    texts = [page.extract_text() or "" for page in reader.pages]
    if any(len(text.strip()) < 100 for text in texts):
        raise ValueError("A nearly empty page requires layout review")
    checks.update({
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": args.source.relative_to(ROOT).as_posix(),
        "source_sha256": digest(args.source),
        "pdf": args.output.relative_to(ROOT).as_posix(),
        "pdf_sha256": digest(args.output),
        "pages": len(reader.pages),
        "page_size": "A4",
        "page_characters": [len(text) for text in texts],
        "heading_pages": doc.heading_pages,
        "visual_review": "pending; rendered pages must be inspected before delivery",
    })
    qa = ROOT / "resultados/local/report_audit/pdf_qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "generation_manifest.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.render:
        render_pdf(args.output, qa)
    print(json.dumps({"pdf": str(args.output), "pages": len(reader.pages), "word_counts": checks["word_counts"], "qa": str(qa)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
