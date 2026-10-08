#!/usr/bin/env python3
"""Render the plain-language DriveOne project guide to a readable PDF.

This renderer is intentionally dependency-light. It handles the Markdown used by
our guide (headings, paragraphs, lists, fenced code, blockquotes, and tables)
with ReportLab, so the PDF can be rebuilt from the tracked source without
putting data or model artifacts in Git.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Iterable

TOOLS = Path(__file__).resolve().parents[1] / ".pdf_tools"
if TOOLS.exists():
    sys.path.insert(0, str(TOOLS))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen.canvas import Canvas
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "DriveOne_Project_Guide.md"
DEFAULT_OUTPUT = ROOT / "reports" / "DriveOne_Project_Guide.pdf"


def register_fonts() -> tuple[str, str]:
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    if all(Path(x).exists() for x in (regular, bold, mono)):
        pdfmetrics.registerFont(TTFont("GuideSans", regular))
        pdfmetrics.registerFont(TTFont("GuideSans-Bold", bold))
        pdfmetrics.registerFont(TTFont("GuideMono", mono))
        return "GuideSans", "GuideMono"
    return "Helvetica", "Courier"


def inline(text: str) -> str:
    """Convert the small Markdown inline subset used by the guide to XML."""
    text = html.unescape(text)
    # Protect code spans before escaping HTML syntax.
    saved: list[str] = []
    def save_code(match: re.Match[str]) -> str:
        saved.append(f'<font name="GuideMono" color="#333333">{escape(match.group(1))}</font>')
        return f"@@@GUIDECODE{len(saved)-1}@@@"
    text = re.sub(r"`([^`]+)`", save_code, text)
    text = escape(text)
    # Markdown links and autolinks.
    text = re.sub(r"&lt;((?:https?://|mailto:)[^&]+)&gt;", r'<link href="\1" color="#1b5e9e">\1</link>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1b5e9e">\1</link>', text)
    # Emphasis after escaping.
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__([^_]+)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    for i, value in enumerate(saved):
        text = text.replace(f"@@@GUIDECODE{i}@@@", value)
    return text


def is_table_line(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def parse_table(lines: list[str], styles: dict[str, ParagraphStyle]):
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return Spacer(1, 1)
    ncols = max(len(row) for row in rows)
    rows = [row + [""] * (ncols - len(row)) for row in rows]
    data = [[Paragraph(inline(cell), styles["TableHeader" if i == 0 else "TableCell"]) for cell in row]
            for i, row in enumerate(rows)]
    # Let long prose columns take more width, while preserving readable numeric columns.
    weights = []
    for col in range(ncols):
        longest = max((len(row[col]) for row in rows), default=10)
        weights.append(max(1.0, min(4.0, longest / 18.0)))
    total = sum(weights)
    width = 170 * mm
    col_widths = [width * w / total for w in weights]
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8f4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#15324b")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aab7c4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
    ]))
    return table


def make_styles(font: str, mono: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle("GuideTitle", parent=base["Title"], fontName=f"{font}-Bold", fontSize=22,
                                 leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#15324b"), spaceAfter=10),
        "Subtitle": ParagraphStyle("GuideSubtitle", parent=base["Normal"], fontName=font, fontSize=10,
                                    leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#52606d"), spaceAfter=16),
        "H2": ParagraphStyle("GuideH2", parent=base["Heading2"], fontName=f"{font}-Bold", fontSize=15,
                              leading=19, textColor=colors.HexColor("#15324b"), spaceBefore=13, spaceAfter=6,
                              keepWithNext=True),
        "H3": ParagraphStyle("GuideH3", parent=base["Heading3"], fontName=f"{font}-Bold", fontSize=11.5,
                              leading=15, textColor=colors.HexColor("#235b7c"), spaceBefore=9, spaceAfter=4,
                              keepWithNext=True),
        "Body": ParagraphStyle("GuideBody", parent=base["BodyText"], fontName=font, fontSize=9.3,
                                leading=13.2, spaceAfter=6, textColor=colors.HexColor("#202a33")),
        "Quote": ParagraphStyle("GuideQuote", parent=base["BodyText"], fontName=font, fontSize=9.1,
                                 leading=13, leftIndent=10, borderPadding=6, borderColor=colors.HexColor("#6a9fbf"),
                                 borderWidth=1, borderLeft=True, textColor=colors.HexColor("#354b5e"), spaceAfter=8),
        "Bullet": ParagraphStyle("GuideBullet", parent=base["BodyText"], fontName=font, fontSize=9.1,
                                  leading=12.5, leftIndent=6, firstLineIndent=0, spaceAfter=2),
        "TableCell": ParagraphStyle("GuideTableCell", parent=base["BodyText"], fontName=font, fontSize=7.5,
                                      leading=9.5, spaceAfter=0),
        "TableHeader": ParagraphStyle("GuideTableHeader", parent=base["BodyText"], fontName=f"{font}-Bold", fontSize=7.5,
                                        leading=9.5, spaceAfter=0),
        "Code": ParagraphStyle("GuideCode", parent=base["Code"], fontName=mono, fontSize=7.1,
                               leading=9, leftIndent=6, rightIndent=6, borderPadding=6,
                               backColor=colors.HexColor("#f1f4f6"), borderColor=colors.HexColor("#d5dde3"),
                               borderWidth=0.5, spaceBefore=3, spaceAfter=8),
    }


def markdown_blocks(source: str, styles: dict[str, ParagraphStyle]) -> list:
    lines = source.replace("\r\n", "\n").split("\n")
    story = []
    i = 0
    first_title = True
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if first_title and line.startswith("# "):
            story.append(Paragraph(inline(line[2:].strip()), styles["Title"]))
            story.append(Paragraph("Plain-language technical record of the DriveOne feasibility project", styles["Subtitle"]))
            first_title = False
            i += 1
            continue
        first_title = False
        if line.startswith(">"):
            q = []
            while i < len(lines) and lines[i].startswith(">"):
                q.append(lines[i].lstrip("> "))
                i += 1
            story.append(Paragraph(inline(" ".join(q)), styles["Quote"]))
            continue
        if line.startswith("## ") or line.startswith("### "):
            level = 2 if line.startswith("## ") and not line.startswith("### ") else 3
            text = line[3 if level == 2 else 4:].strip()
            story.append(Paragraph(inline(text), styles["H2" if level == 2 else "H3"]))
            i += 1
            continue
        if line.strip() == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#b8c4ce"), spaceBefore=4, spaceAfter=8))
            i += 1
            continue
        if line.strip().startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            story.append(Preformatted("\n".join(code), styles["Code"], maxLineLength=120))
            continue
        if is_table_line(line):
            table_lines = []
            while i < len(lines) and is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            story.append(parse_table(table_lines, styles))
            story.append(Spacer(1, 5))
            continue
        if re.match(r"^\s*[-*] ", line) or re.match(r"^\s*\d+[.)] ", line):
            ordered = bool(re.match(r"^\s*\d+[.)] ", line))
            items = []
            while i < len(lines):
                m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*)$", lines[i])
                if not m:
                    break
                text = m.group(1)
                i += 1
                # Continuation lines are folded into the same item.
                cont = []
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                    cont.append(lines[i].strip())
                    i += 1
                if cont:
                    text += " " + " ".join(cont)
                items.append(ListItem(Paragraph(inline(text), styles["Bullet"]), leftIndent=9))
            story.append(ListFlowable(items, bulletType="1" if ordered else "bullet", start="1", leftIndent=14,
                                      bulletFontName=styles["Bullet"].fontName, bulletFontSize=8.5))
            story.append(Spacer(1, 4))
            continue
        # Paragraph: consume ordinary lines until the next block marker.
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip():
            nxt = lines[i]
            if (nxt.startswith(("## ", "### ", ">")) or nxt.strip().startswith("```") or
                    nxt.strip() == "---" or is_table_line(nxt) or
                    re.match(r"^\s*[-*] ", nxt) or re.match(r"^\s*\d+[.)] ", nxt)):
                break
            para.append(nxt.strip())
            i += 1
        story.append(Paragraph(inline(" ".join(para)), styles["Body"]))
    return story


class NumberedCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(page_count)
            Canvas.showPage(self)
        Canvas.save(self)

    def draw_page_number(self, page_count: int):
        width, height = A4
        self.saveState()
        self.setStrokeColor(colors.HexColor("#c7d0d8"))
        self.setLineWidth(0.35)
        self.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
        self.setFont("GuideSans", 7.5)
        self.setFillColor(colors.HexColor("#61717e"))
        self.drawString(18 * mm, 8 * mm, "DriveOne Project Guide — technical record")
        self.drawRightString(width - 18 * mm, 8 * mm, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build(source: Path, output: Path) -> None:
    font, mono = register_fonts()
    styles = make_styles(font, mono)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(output), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="DriveOne Project Guide", author="DriveOne project",
        subject="Plain-language technical record of the DriveOne feasibility project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="guide", frames=[frame])])
    story = markdown_blocks(source.read_text(), styles)
    doc.build(story, canvasmaker=NumberedCanvas)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    build(src, out)
    print(out)
