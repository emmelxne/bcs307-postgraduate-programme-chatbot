"""Step 1 – Prepare the handbook for NLP.

Reads the .docx handbook in document order (paragraphs AND tables), keeps the
heading hierarchy, turns every table row into a self-describing sentence, and
splits each section into retrieval chunks. Output: data/processed/chunks.json

Run:  python src/ingest.py
"""
import json
import re
import sys
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

sys.path.insert(0, str(Path(__file__).parent))
from programmes import detect_in_heading, detect_mode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "FCDT_Postgraduate_Handbook_2026-2027.docx"
OUT = ROOT / "data" / "processed" / "chunks.json"
MAX_WORDS = 160  # chunk size limit


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def iter_blocks(document):
    """Yield Paragraph and Table objects in the order they appear."""
    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            yield Paragraph(child, document)
        elif tag == "tbl":
            yield Table(child, document)


def table_to_units(table: Table) -> list[str]:
    """Convert a table into one sentence per row: 'Header: value; Header: value'."""
    rows = []
    for row in table.rows:
        # Merged cells repeat the same underlying element; keep the value only in
        # the LAST column it spans so totals line up with the 'Credit' column.
        tcs = [c._tc for c in row.cells]
        cells = []
        for i, cell in enumerate(row.cells):
            nxt_same = i + 1 < len(tcs) and tcs[i + 1] is tcs[i]
            prev_same = i > 0 and tcs[i - 1] is tcs[i]
            if nxt_same:
                cells.append("")  # placeholder; value emitted at end of span
            else:
                txt = clean(cell.text)
                if prev_same:  # end of a merged span: label = text, put in last col
                    cells[-1] = ""
                cells.append(txt)
        rows.append(cells)
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    units, carry = [], {}
    for raw in body:
        cells = list(raw)
        if not any(cells):
            continue
        # A row whose first non-empty cell is a label like 'Total credit hours ...'
        label = next((c for c in cells[:-1] if c), "")
        if label and re.match(r"(?i)(sub ?total|total)", label):
            units.append(f"{label}: {cells[-1]}.")
            continue
        parts = []
        for i, val in enumerate(cells):
            col = header[i] if i < len(header) else f"Col{i + 1}"
            # Carry forward blank 'Semester' cells (one semester spans several rows)
            if "semester" in col.lower():
                if val:
                    carry[i] = val
                else:
                    val = carry.get(i, "")
            if val:
                parts.append(f"{col}: {val}" if col else val)
        if parts:
            units.append("; ".join(parts) + ".")
    return units


def build_sections(path: Path):
    document = docx.Document(str(path))
    sections = []
    stack: list[str] = []  # heading path
    current = {"path": ["Front Matter"], "units": []}
    in_toc = False

    for block in iter_blocks(document):
        if isinstance(block, Paragraph):
            text = clean(block.text)
            style = block.style.name if block.style is not None else ""
            if not text:
                continue
            if text == "Table of Contents":
                in_toc = True
                continue
            m = re.match(r"Heading (\d)", style)
            if m:
                in_toc = False
                level = int(m.group(1))
                if current["units"]:
                    sections.append(current)
                stack = stack[: level - 1] + [text]
                current = {"path": list(stack), "units": []}
                continue
            if in_toc:
                continue
            current["units"].append(text)
        else:
            if in_toc:
                continue
            current["units"].extend(table_to_units(block))
    if current["units"]:
        sections.append(current)
    return sections


def chunk_sections(sections):
    chunks = []
    for sec in sections:
        path_text = " > ".join(sec["path"])
        programmes = set()
        for h in sec["path"]:
            programmes |= detect_in_heading(h)
        mode = detect_mode(sec["path"][-1])
        buf, words = [], 0
        for unit in sec["units"]:
            n = len(unit.split())
            if buf and words + n > MAX_WORDS:
                chunks.append((path_text, programmes, mode, buf))
                buf, words = [], 0
            buf.append(unit)
            words += n
        if buf:
            chunks.append((path_text, programmes, mode, buf))

    out = []
    for i, (path_text, programmes, mode, units) in enumerate(chunks):
        out.append({
            "id": i,
            "section": path_text,
            "programmes": sorted(programmes),
            "mode": mode,
            "units": units,                       # sentences / table rows
            "text": path_text + ". " + " ".join(units),  # what gets indexed
        })
    return out


def main():
    sections = build_sections(RAW)
    chunks = chunk_sections(sections)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    words = sum(len(c["text"].split()) for c in chunks)
    print(f"Sections: {len(sections)} | Chunks: {len(chunks)} | Words indexed: {words}")
    print(f"Saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
