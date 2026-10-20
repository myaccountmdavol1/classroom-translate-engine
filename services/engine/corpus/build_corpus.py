"""Generate a corpus of realistic educational PDFs for testing.

These stand in for real student documents (which must never be committed to a
repository). Each one targets a specific failure mode the engine has to
survive: rotated pages, rasterized scans, diagram labels, dense tables,
answer-choice layouts, and long documents.

Run:  python corpus/build_corpus.py
Out:  corpus/out/*.pdf
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as rl_canvas

OUT = Path(__file__).parent / "out"
PAGE_W, PAGE_H = letter  # 612 x 792 points

INK = HexColor("#111111")
GREY = HexColor("#666666")
RULE = HexColor("#999999")


# --------------------------------------------------------------------------
# Synthetic images
# --------------------------------------------------------------------------

# Real educational diagrams carry anti-aliased labels at a readable size.
# PIL's built-in bitmap font is ~6x11 and far worse than anything a teacher
# would actually hand out, so using it everywhere would make the image-text
# corpus unrepresentatively hard. A genuinely low-quality case is generated
# separately as 15_lowres_diagram.pdf.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def label_font(size: int):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def cell_diagram_png(w: int = 1100, h: int = 820, label_size: int = 26,
                     font=None) -> bytes:
    """A labeled biology diagram -- the phase-4 image-text target."""
    sx, sy = w / 620.0, h / 460.0
    font = font or label_font(label_size)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([60 * sx, 40 * sy, w - 60 * sx, h - 40 * sy],
              outline="#2b6b3f", width=max(2, int(5 * sx)), fill="#eef7f0")
    d.ellipse([230 * sx, 170 * sy, 390 * sx, 300 * sy],
              outline="#8a3b8a", width=max(2, int(4 * sx)), fill="#f3e6f3")
    d.ellipse([285 * sx, 215 * sy, 335 * sx, 255 * sy],
              outline="#8a3b8a", width=max(2, int(3 * sx)), fill="#d9b8d9")
    for cx, cy in ((150, 140), (460, 320), (180, 340)):
        d.ellipse([(cx - 28) * sx, (cy - 16) * sy, (cx + 28) * sx, (cy + 16) * sy],
                  outline="#b8860b", width=max(2, int(3 * sx)))
    # Labels with leader lines -- text embedded in a raster image. Leader
    # lines start clear of the text so they do not strike through a glyph.
    labels = [
        ("Cell Wall", (14, 16), (95, 62)),
        ("Nucleus", (452, 150), (392, 205)),
        ("Nucleolus", (440, 420), (330, 262)),
        ("Mitochondrion", (10, 408), (168, 352)),
        ("Chloroplast", (440, 38), (474, 310)),
    ]
    for text, (tx, ty), (px, py) in labels:
        x, y = tx * sx, ty * sy
        d.text((x, y), text, fill="#111111", font=font)
        box = d.textbbox((x, y), text, font=font)
        d.line([box[0] + (box[2] - box[0]) / 2, box[3] + 3 * sy, px * sx, py * sy],
               fill="#555555", width=max(1, int(2 * sx)))
    d.text((14 * sx, h - 34 * sy), "Figure 3: Plant Cell Structure",
           fill="#333333", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def photograph_png(w: int = 520, h: int = 360) -> bytes:
    """A photo-like raster: continuous tone, no text. Must be preserved as-is."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            r = int(90 + 80 * math.sin(x / 55.0) + 40 * math.cos(y / 40.0))
            g = int(120 + 60 * math.sin((x + y) / 70.0))
            b = int(150 + 70 * math.cos(x / 90.0))
            px[x, y] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    d = ImageDraw.Draw(img)
    d.ellipse([w * 0.6, h * 0.1, w * 0.85, h * 0.35], fill="#fdf6d8")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def graph_png(w: int = 900, h: int = 675, label_size: int = 22) -> bytes:
    """A line chart with axis labels -- text inside a graphic."""
    f = label_font(label_size)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    left, top, bottom = w * 0.14, h * 0.09, h - h * 0.17
    d.rectangle([left, top, w - 20, bottom], outline="#333333", width=2)
    for i in range(1, 5):
        y = top + i * (bottom - top) / 5
        d.line([left, y, w - 20, y], fill="#dddddd", width=1)
    pts = [(left + i * (w - left - 30) / 8,
            bottom - (30 + 25 * math.sin(i / 1.4)) * (h / 360.0) * 1.6)
           for i in range(9)]
    d.line(pts, fill="#1f5fa8", width=3)
    d.text((8, h / 2 - label_size), "Temp", fill="#111111", font=f)
    d.text((8, h / 2 + 2), "(°C)", fill="#111111", font=f)
    d.text((w / 2 - 70, bottom + 22), "Time (hours)", fill="#111111", font=f)
    d.text((left + 20, 6), "Heating Curve of Water", fill="#111111", font=f)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def handwriting_png(w: int = 460, h: int = 110) -> bytes:
    """Wobbly strokes standing in for a handwritten student answer."""
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    x = 20.0
    while x < w - 30:
        amp = 12 + 8 * math.sin(x / 17.0)
        pts = [(x + t, 60 + amp * math.sin((x + t) / 9.0) - t * 0.05)
               for t in range(0, 22, 2)]
        d.line(pts, fill="#22337a", width=2)
        x += 26
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def header(c, title: str, subtitle: str | None = None) -> float:
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, PAGE_H - 0.9 * inch, title)
    y = PAGE_H - 1.15 * inch
    if subtitle:
        c.setFont("Helvetica", 10)
        c.setFillColor(GREY)
        c.drawString(1 * inch, y, subtitle)
        y -= 0.22 * inch
    c.setStrokeColor(RULE)
    c.setLineWidth(0.8)
    c.line(1 * inch, y, PAGE_W - 1 * inch, y)
    c.setFillColor(INK)
    return y - 0.3 * inch


def name_block(c) -> None:
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, PAGE_H - 0.55 * inch, "Name: ______________________")
    c.drawString(4.6 * inch, PAGE_H - 0.55 * inch, "Date: ______________")
    c.drawString(6.6 * inch, PAGE_H - 0.55 * inch, "Period: ______")


def wrapped(c, text: str, x: float, y: float, width: float,
            font: str = "Helvetica", size: float = 10.5, leading: float = 14.0) -> float:
    c.setFont(font, size)
    words, line = text.split(), ""
    for w in words:
        trial = f"{line} {w}".strip()
        if c.stringWidth(trial, font, size) <= width:
            line = trial
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def answer_lines(c, x: float, y: float, width: float, count: int = 4,
                 gap: float = 22.0) -> float:
    c.setStrokeColor(RULE)
    c.setLineWidth(0.7)
    for _ in range(count):
        c.line(x, y, x + width, y)
        y -= gap
    c.setStrokeColor(INK)
    return y


def footer(c, page_no: int, label: str = "") -> None:
    c.setFont("Helvetica", 8.5)
    c.setFillColor(GREY)
    if label:
        c.drawString(1 * inch, 0.55 * inch, label)
    c.drawCentredString(PAGE_W / 2, 0.55 * inch, str(page_no))
    c.setFillColor(INK)


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
def doc_clean_worksheet(path: Path) -> None:
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Photosynthesis Review Worksheet")
    c.setAuthor("Science Department")
    name_block(c)
    y = header(c, "Photosynthesis Review", "Living Environment - Unit 4")
    y = wrapped(c, "Directions: Read each question carefully. Write your answers "
                   "in complete sentences on the lines provided.",
                1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10.5)
    y -= 12
    for n, q in enumerate(
        ["Explain why chlorophyll is necessary for photosynthesis.",
         "Describe what happens to the glucose a plant produces.",
         "A plant is placed in a dark closet for one week. Predict what will "
         "happen to its leaves and explain your reasoning."], start=1):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1 * inch, y, f"{n}.")
        y = wrapped(c, q, 1.3 * inch, y, 6.2 * inch, "Helvetica", 11)
        y -= 6
        y = answer_lines(c, 1.3 * inch, y, 6.2 * inch, 3)
        y -= 14
    footer(c, 1, "Photosynthesis Review")
    c.save()


def doc_math_worksheet(path: Path) -> None:
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Algebra I - Solving Equations")
    name_block(c)
    y = header(c, "Solving Linear Equations", "Algebra I - Practice Set B")
    y = wrapped(c, "Directions: Solve each equation. Show all of your work in the "
                   "space provided. Circle your final answer.",
                1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10.5)
    y -= 10
    problems = ["3x + 7 = 22", "5(y - 2) = 35", "2a/3 + 4 = 10",
                "-4m + 9 = -7", "0.5t + 1.25 = 4.75", "x^2 - 9 = 0"]
    col_x = [1 * inch, 4.3 * inch]
    start_y = y
    for i, p in enumerate(problems):
        cx = col_x[i % 2]
        cy = start_y - (i // 2) * 1.55 * inch
        c.setFont("Helvetica-Bold", 11)
        c.drawString(cx, cy, f"{i + 1}.")
        c.setFont("Times-Roman", 12)
        c.drawString(cx + 0.25 * inch, cy, p)
        c.setStrokeColor(RULE)
        c.rect(cx + 0.25 * inch, cy - 1.15 * inch, 2.5 * inch, 1.0 * inch, stroke=1, fill=0)
        c.setStrokeColor(INK)
    y = start_y - 3 * 1.55 * inch - 10
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(1 * inch, y, "Formula Reference")
    y -= 14
    c.setFont("Times-Roman", 11)
    for f in ["A = πr²", "H₂O", "E = mc²", "a² + b² = c²"]:
        c.drawString(1.2 * inch, y, f)
        y -= 15
    footer(c, 1, "Algebra I")
    c.save()


def doc_science_diagram(path: Path) -> None:
    from reportlab.lib.utils import ImageReader
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Plant Cell Structure")
    name_block(c)
    y = header(c, "Plant Cell Structure", "Living Environment - Lab 7")
    y = wrapped(c, "Directions: Study the diagram below. Then answer the questions "
                   "that follow using evidence from the diagram.",
                1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10.5)
    y -= 8
    img = ImageReader(io.BytesIO(cell_diagram_png()))
    ih = 3.0 * inch
    iw = ih * (620 / 460)
    c.drawImage(img, 1 * inch, y - ih, width=iw, height=ih)
    y -= ih + 18
    for n, q in enumerate(["Which structure controls the activities of the cell?",
                           "Explain the function of the chloroplast."], start=1):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1 * inch, y, f"{n}.")
        y = wrapped(c, q, 1.3 * inch, y, 6.2 * inch, "Helvetica", 11)
        y -= 4
        y = answer_lines(c, 1.3 * inch, y, 6.2 * inch, 2)
        y -= 10
    footer(c, 1, "Lab 7")
    c.save()


def doc_exam_multiple_choice(path: Path, pages: int = 4) -> None:
    from reportlab.lib.utils import ImageReader
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Regents-Style Examination in Living Environment")
    questions = [
        ("Which organelle is the site of cellular respiration?",
         ["ribosome", "mitochondrion", "vacuole", "cell wall"]),
        ("A student observes bubbles forming on a water plant in sunlight. "
         "The gas is most likely",
         ["carbon dioxide", "nitrogen", "oxygen", "water vapor"]),
        ("Which process produces genetically identical daughter cells?",
         ["meiosis", "mitosis", "fertilization", "mutation"]),
        ("The main function of the cell membrane is to",
         ["store genetic material", "produce energy",
          "regulate what enters and leaves", "synthesize proteins"]),
        ("Which of these is an abiotic factor in an ecosystem?",
         ["decomposers", "temperature", "producers", "consumers"]),
    ]
    qnum = 1
    for pg in range(pages):
        if pg == 0:
            c.setFont("Helvetica-Bold", 14)
            c.drawCentredString(PAGE_W / 2, PAGE_H - 0.8 * inch,
                                "The University of the State of New York")
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(PAGE_W / 2, PAGE_H - 1.02 * inch,
                                "REGENTS HIGH SCHOOL EXAMINATION")
            c.setFont("Helvetica", 11)
            c.drawCentredString(PAGE_W / 2, PAGE_H - 1.24 * inch,
                                "LIVING ENVIRONMENT")
            y = PAGE_H - 1.7 * inch
            y = wrapped(c, "Part A - Answer all questions in this part. For each "
                           "statement or question, record on your separate answer "
                           "sheet the number of the word or expression that best "
                           "completes the statement or answers the question.",
                        1 * inch, y, 6.5 * inch, "Helvetica", 10.5)
            y -= 14
        else:
            y = PAGE_H - 0.9 * inch
        if pg == 2:
            img = ImageReader(io.BytesIO(graph_png()))
            c.drawImage(img, 1.4 * inch, y - 2.6 * inch, width=3.9 * inch, height=2.6 * inch)
            y -= 2.6 * inch + 14
            y = wrapped(c, "Base your answers to the following questions on the "
                           "graph above and on your knowledge of science.",
                        1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10)
            y -= 10
        for q, choices in questions:
            if y < 2.0 * inch:
                break
            c.setFont("Helvetica", 11)
            c.drawString(1 * inch, y, f"{qnum}")
            y = wrapped(c, q, 1.35 * inch, y, 6.0 * inch, "Helvetica", 11)
            y -= 2
            for li, ch in enumerate(choices):
                c.setFont("Helvetica", 10.5)
                c.drawString(1.5 * inch, y, f"({li + 1})")
                c.drawString(1.95 * inch, y, ch)
                y -= 14
            y -= 8
            qnum += 1
        footer(c, pg + 1, "Living Environment")
        c.showPage()
    c.save()


def doc_table_document(path: Path) -> None:
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Data Analysis: Population Study")
    name_block(c)
    y = header(c, "Data Analysis", "Population Study - Table 1")
    y = wrapped(c, "Directions: Use the data table below to answer the questions.",
                1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10.5)
    y -= 16
    cols = ["Year", "Population", "Births", "Deaths", "Net Change"]
    rows = [["2018", "12,400", "310", "185", "+125"],
            ["2019", "12,525", "298", "192", "+106"],
            ["2020", "12,631", "265", "240", "+25"],
            ["2021", "12,656", "288", "201", "+87"],
            ["2022", "12,743", "301", "188", "+113"]]
    widths = [0.9, 1.5, 1.1, 1.1, 1.3]
    x0 = 1 * inch
    rh = 20
    c.setFont("Helvetica-Bold", 10)
    cx = x0
    for cidx, col in enumerate(cols):
        w = widths[cidx] * inch
        c.setStrokeColor(RULE)
        c.rect(cx, y - rh, w, rh, stroke=1, fill=0)
        c.setFillColor(INK)
        c.drawString(cx + 5, y - rh + 6, col)
        cx += w
    y -= rh
    c.setFont("Helvetica", 10)
    for row in rows:
        cx = x0
        for cidx, cell in enumerate(row):
            w = widths[cidx] * inch
            c.rect(cx, y - rh, w, rh, stroke=1, fill=0)
            c.drawString(cx + 5, y - rh + 6, cell)
            cx += w
        y -= rh
    c.setStrokeColor(INK)
    y -= 24
    for n, q in enumerate(["In which year was the net change the smallest?",
                           "Calculate the total number of births from 2018 to 2022."],
                          start=1):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1 * inch, y, f"{n}.")
        y = wrapped(c, q, 1.3 * inch, y, 6.2 * inch, "Helvetica", 11)
        y = answer_lines(c, 1.3 * inch, y, 6.2 * inch, 2)
        y -= 12
    footer(c, 1, "Data Analysis")
    c.save()


def doc_photo_document(path: Path) -> None:
    from reportlab.lib.utils import ImageReader
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Landforms Photo Study")
    y = header(c, "Landforms of the United States", "Grade 5 Social Studies")
    y = wrapped(c, "Directions: Look closely at the photograph. Write three "
                   "observations about the landform you see.",
                1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10.5)
    y -= 10
    img = ImageReader(io.BytesIO(photograph_png()))
    c.drawImage(img, 1 * inch, y - 2.8 * inch, width=4.0 * inch, height=2.8 * inch)
    y -= 2.8 * inch + 8
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(GREY)
    c.drawString(1 * inch, y, "Photograph 1: A coastal region at sunrise.")
    c.setFillColor(INK)
    y -= 26
    y = answer_lines(c, 1 * inch, y, 6.5 * inch, 6)
    footer(c, 1, "Social Studies")
    c.save()


def doc_reading_passage(path: Path, pages: int = 3) -> None:
    body = ("The monarch butterfly begins its life as a tiny egg attached to the "
            "underside of a milkweed leaf. Within a few days, a caterpillar emerges "
            "and begins to eat. Milkweed is the only plant a monarch caterpillar "
            "will eat, and it contains a chemical that makes the caterpillar taste "
            "bitter to birds. As the caterpillar grows, it sheds its skin five "
            "separate times. Each stage between sheddings is called an instar. ")
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("The Journey of the Monarch")
    for pg in range(pages):
        if pg == 0:
            y = header(c, "The Journey of the Monarch", "Grade 6 Reading Passage")
        else:
            y = PAGE_H - 0.9 * inch
            c.setFont("Helvetica-Oblique", 9)
            c.setFillColor(GREY)
            c.drawString(1 * inch, y, "The Journey of the Monarch (continued)")
            c.setFillColor(INK)
            y -= 20
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1 * inch, y, f"Section {pg + 1}")
        y -= 18
        for para in range(4):
            y = wrapped(c, body * 2, 1 * inch, y, 6.5 * inch, "Times-Roman", 11, 15)
            y -= 10
            if y < 1.4 * inch:
                break
        footer(c, pg + 1, "Reading")
        c.showPage()
    c.save()


def doc_mixed_orientation(path: Path) -> None:
    from reportlab.lib.utils import ImageReader
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Mixed Orientation Packet")
    y = header(c, "Unit Review Packet", "Page 1 - Portrait")
    y = wrapped(c, "Directions: Complete both sides of this packet.",
                1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10.5)
    answer_lines(c, 1 * inch, y - 20, 6.5 * inch, 5)
    footer(c, 1)
    c.showPage()

    # Landscape page created by swapping the page size -- the common case.
    c.setPageSize(landscape(letter))
    lw, lh = landscape(letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, lh - 0.9 * inch, "Wide Data Table (Landscape)")
    img = ImageReader(io.BytesIO(graph_png()))
    c.drawImage(img, 1 * inch, lh - 4.2 * inch, width=4.6 * inch, height=3.0 * inch)
    c.setFont("Helvetica", 10.5)
    c.drawString(6.2 * inch, lh - 1.6 * inch, "Analyze the heating curve.")
    c.setFont("Helvetica", 9)
    c.drawCentredString(lw / 2, 0.55 * inch, "2")
    c.showPage()

    c.setPageSize(letter)
    y = header(c, "Reflection", "Page 3 - Portrait")
    answer_lines(c, 1 * inch, y - 10, 6.5 * inch, 8)
    footer(c, 3)
    c.save()


def doc_handwriting(path: Path) -> None:
    from reportlab.lib.utils import ImageReader
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Student Work Sample")
    y = header(c, "Student Work Sample", "Exit Ticket")
    c.setFont("Helvetica", 11)
    c.drawString(1 * inch, y, "1. Explain how you solved the problem.")
    y -= 24
    img = ImageReader(io.BytesIO(handwriting_png()))
    c.drawImage(img, 1.2 * inch, y - 0.85 * inch, width=4.6 * inch, height=1.1 * inch)
    y -= 1.2 * inch
    c.setFont("Helvetica", 11)
    c.drawString(1 * inch, y, "2. What would you do differently next time?")
    y -= 24
    c.drawImage(img, 1.2 * inch, y - 0.85 * inch, width=4.6 * inch, height=1.1 * inch)
    footer(c, 1, "Exit Ticket")
    c.save()


def doc_long(path: Path, pages: int) -> None:
    """A long packet: mixed content, real page count, no filler-only pages."""
    from reportlab.lib.utils import ImageReader
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle(f"Comprehensive Review Packet ({pages} pages)")
    diagram = ImageReader(io.BytesIO(cell_diagram_png()))
    chart = ImageReader(io.BytesIO(graph_png()))
    passage = ("Scientists studying coastal erosion collected measurements over a "
               "ten year period. The data showed that shorelines protected by dune "
               "grass eroded more slowly than unprotected shorelines. ")
    qnum = 1
    for pg in range(pages):
        kind = pg % 5
        if pg == 0:
            name_block(c)
            y = header(c, "Comprehensive Review Packet", "All Units")
        else:
            y = PAGE_H - 0.85 * inch
            c.setFont("Helvetica-Oblique", 9)
            c.setFillColor(GREY)
            c.drawString(1 * inch, y, "Comprehensive Review Packet")
            c.setFillColor(INK)
            y -= 22

        if kind == 0:
            y = wrapped(c, "Directions: Answer each question in complete sentences.",
                        1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10.5)
            y -= 8
            for _ in range(3):
                c.setFont("Helvetica-Bold", 11)
                c.drawString(1 * inch, y, f"{qnum}.")
                y = wrapped(c, "Describe one way that organisms in an ecosystem "
                               "depend on each other for survival.",
                            1.3 * inch, y, 6.2 * inch, "Helvetica", 11)
                y = answer_lines(c, 1.3 * inch, y - 4, 6.2 * inch, 3)
                y -= 12
                qnum += 1
        elif kind == 1:
            for _ in range(3):
                y = wrapped(c, passage * 3, 1 * inch, y, 6.5 * inch, "Times-Roman", 11, 15)
                y -= 10
        elif kind == 2:
            c.drawImage(diagram, 1.1 * inch, y - 2.7 * inch, width=3.64 * inch, height=2.7 * inch)
            y -= 2.9 * inch
            c.setFont("Helvetica-Bold", 11)
            c.drawString(1 * inch, y, f"{qnum}.")
            y = wrapped(c, "Label the structure indicated and explain its function.",
                        1.3 * inch, y, 6.2 * inch, "Helvetica", 11)
            y = answer_lines(c, 1.3 * inch, y - 4, 6.2 * inch, 4)
            qnum += 1
        elif kind == 3:
            for _ in range(4):
                c.setFont("Helvetica", 11)
                c.drawString(1 * inch, y, f"{qnum}")
                y = wrapped(c, "Which statement best describes the relationship "
                               "shown in the data?", 1.35 * inch, y, 6.0 * inch,
                            "Helvetica", 11)
                for li, ch in enumerate(["It increases steadily.", "It decreases steadily.",
                                         "It stays the same.", "It varies randomly."]):
                    c.setFont("Helvetica", 10.5)
                    c.drawString(1.5 * inch, y, f"({li + 1})")
                    c.drawString(1.95 * inch, y, ch)
                    y -= 14
                y -= 8
                qnum += 1
        else:
            c.drawImage(chart, 1.3 * inch, y - 2.5 * inch, width=3.75 * inch, height=2.5 * inch)
            y -= 2.7 * inch
            y = wrapped(c, "Base your answers on the graph above.",
                        1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10)
            y = answer_lines(c, 1 * inch, y - 8, 6.5 * inch, 5)
        footer(c, pg + 1, "Review Packet")
        c.showPage()
    c.save()


def doc_lowres_diagram(path: Path) -> None:
    """A diagram whose labels are genuinely hard to read.

    Small bitmap type at low resolution, which is what a third-generation
    photocopy of a textbook figure looks like. The engine must decline to
    replace labels it cannot read confidently rather than guessing.
    """
    from reportlab.lib.utils import ImageReader

    tiny = ImageFont.load_default()
    png = cell_diagram_png(w=430, h=320, font=tiny)
    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Low Resolution Diagram")
    y = header(c, "Cell Structures", "Photocopied Figure")
    y = wrapped(c, "Directions: Label the structures shown in the figure.",
                1 * inch, y, 6.5 * inch, "Helvetica-Oblique", 10.5)
    img = ImageReader(io.BytesIO(png))
    c.drawImage(img, 1 * inch, y - 2.6 * inch, width=3.5 * inch, height=2.6 * inch)
    footer(c, 1, "Figure")
    c.save()


def tilted_labels_png(w: int = 1400, h: int = 900) -> bytes:
    """Labels printed at a slant inside one picture.

    Level, 8, 15 and 25 degrees; a title around an arc; white lettering on
    a navy panel; and a label on a tilted "tablet" whose dark frame runs
    just above it -- the shape of the real slide where a tilted word's
    upright box took in the artwork around it.
    """
    img = Image.new("RGB", (w, h), "white")
    font = label_font(34)

    def tilted(text, angle, centre, *, fill="#111111", panel=None):
        l, t, r, b = font.getbbox(text)
        tile = Image.new("RGBA", (r - l + 40, b - t + 40), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        if panel:
            d.rectangle([0, 0, tile.width, tile.height], fill=panel)
        d.text((20 - l, 20 - t), text, font=font, fill=fill)
        tile = tile.rotate(-angle, expand=True, resample=Image.BICUBIC)
        img.paste(tile, (int(centre[0] - tile.width / 2), int(centre[1] - tile.height / 2)), tile)

    tilted("Water Cycle", 0, (260, 90))
    tilted("Evaporation", 8, (300, 260))
    tilted("Condensation", 15, (330, 450))
    tilted("Precipitation", 25, (360, 680))
    tilted("Collection Pond", 12, (1050, 760), fill="white", panel="#14215a")

    # A tablet held at a slant, its frame just above the label.
    angle = 14
    tablet = Image.new("RGBA", (420, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(tablet)
    d.rounded_rectangle([0, 0, 419, 299], radius=30, fill="#111111")
    d.rectangle([30, 30, 389, 269], fill="white")
    l, t, r, b = font.getbbox("Science Lab")
    d.text((210 - (r - l) / 2 - l, 42 - t), "Science Lab", font=font, fill="#111111")
    d.rectangle([90, 120, 330, 240], fill="#b02a2a")
    tablet = tablet.rotate(-angle, expand=True, resample=Image.BICUBIC)
    img.paste(tablet, (820, 120), tablet)

    # A title around an arc, letter by letter.
    title, radius, arc = "OUR CLASSROOM", 520, 44
    cx, cy = 1000, 80 + radius
    for i, ch in enumerate(title):
        if ch == " ":
            continue
        a = math.radians(-arc / 2 + arc * i / (len(title) - 1))
        x, y = cx + radius * math.sin(a), cy - radius * math.cos(a)
        tilted(ch, math.degrees(a), (x, y))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def doc_tilted_labels(path: Path) -> None:
    from reportlab.lib.utils import ImageReader

    c = rl_canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Tilted Labels")
    y = header(c, "The Water Cycle", "Labels at an angle")
    img = ImageReader(io.BytesIO(tilted_labels_png()))
    iw = 6.5 * inch
    ih = iw * 900 / 1400
    c.drawImage(img, 1 * inch, y - ih - 10, width=iw, height=ih)
    footer(c, 1, "Figure")
    c.save()


def doc_scanned(src: Path, out: Path, dpi: int = 150) -> None:
    """Rasterize an existing PDF so it has no text layer at all.

    This is what a photocopied worksheet run through a school scanner looks
    like to the engine: pure pixels, OCR required.
    """
    import fitz
    src_doc = fitz.open(str(src))
    out_doc = fitz.open()
    for page in src_doc:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        # Light speckle so it reads like a real scan rather than a clean render.
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        import random
        rnd = random.Random(7)
        px = img.load()
        for _ in range(int(img.width * img.height * 0.002)):
            x = rnd.randrange(img.width)
            y = rnd.randrange(img.height)
            px[x, y] = max(0, px[x, y] - rnd.randrange(40, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        new = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    out_doc.save(str(out))
    out_doc.close()
    src_doc.close()


def doc_rotated(out: Path) -> None:
    """A packet whose landscape pages use /Rotate rather than a swapped size.

    Scanners and Word both produce these, and they are the case that breaks
    naive coordinate handling.
    """
    import fitz
    tmp = OUT / "_rot_src.pdf"
    doc_mixed_orientation(tmp)
    d = fitz.open(str(tmp))
    page2 = d[1]
    page2.set_rotation(90)
    extra = d.new_page(width=612, height=792)
    extra.insert_text((72, 120), "Rotated Answer Sheet", fontsize=16)
    extra.insert_text((72, 160), "Fill in one circle per question.", fontsize=11)
    for i in range(6):
        y = 200 + i * 40
        extra.insert_text((72, y), f"{i + 1}.", fontsize=11)
        for j in range(4):
            extra.draw_circle(fitz.Point(120 + j * 60, y - 4), 7, color=(0, 0, 0), width=1)
    extra.set_rotation(270)
    d.save(str(out))
    d.close()
    tmp.unlink(missing_ok=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing corpus to {OUT}")
    jobs = [
        ("01_clean_worksheet.pdf", lambda p: doc_clean_worksheet(p)),
        ("03_math_worksheet.pdf", lambda p: doc_math_worksheet(p)),
        ("04_science_diagram.pdf", lambda p: doc_science_diagram(p)),
        ("05_tables.pdf", lambda p: doc_table_document(p)),
        ("06_photographs.pdf", lambda p: doc_photo_document(p)),
        ("07_reading_passage.pdf", lambda p: doc_reading_passage(p, 3)),
        ("08_exam_multiple_choice.pdf", lambda p: doc_exam_multiple_choice(p, 4)),
        ("09_handwriting.pdf", lambda p: doc_handwriting(p)),
        ("10_mixed_orientation.pdf", lambda p: doc_mixed_orientation(p)),
        ("11_long_50p.pdf", lambda p: doc_long(p, 52)),
        ("12_long_100p.pdf", lambda p: doc_long(p, 100)),
        ("15_lowres_diagram.pdf", lambda p: doc_lowres_diagram(p)),
        ("17_tilted_labels.pdf", lambda p: doc_tilted_labels(p)),
    ]
    for name, fn in jobs:
        path = OUT / name
        fn(path)
        print(f"  {name:32s} {path.stat().st_size / 1024:8.1f} KB")

    doc_rotated(OUT / "13_rotated_pages.pdf")
    print(f"  {'13_rotated_pages.pdf':32s} "
          f"{(OUT / '13_rotated_pages.pdf').stat().st_size / 1024:8.1f} KB")

    doc_scanned(OUT / "01_clean_worksheet.pdf", OUT / "02_scanned_worksheet.pdf")
    print(f"  {'02_scanned_worksheet.pdf':32s} "
          f"{(OUT / '02_scanned_worksheet.pdf').stat().st_size / 1024:8.1f} KB")
    doc_scanned(OUT / "08_exam_multiple_choice.pdf", OUT / "14_scanned_exam.pdf")
    print(f"  {'14_scanned_exam.pdf':32s} "
          f"{(OUT / '14_scanned_exam.pdf').stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
