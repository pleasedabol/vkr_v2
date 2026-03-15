import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_docx_text(docx_path: Path) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    lines: list[str] = []
    for para in root.findall(".//w:p", ns):
        parts = [t.text for t in para.findall(".//w:t", ns) if t.text]
        lines.append("".join(parts).rstrip())
    out = "\n".join(lines)
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/extract_docx_text.py <file.docx> [out.txt]")
        return 2
    docx = Path(sys.argv[1])
    if not docx.exists():
        # Work around Windows/PowerShell Unicode quoting issues (e.g., long dash in filename).
        # Try to resolve by searching in the current directory.
        candidates = list(Path(".").glob("*.docx"))
        # Prefer an updated copy if present
        updated = [c for c in candidates if c.name.lower().endswith(".updated.docx")]
        if updated:
            docx = sorted(updated, key=lambda p: p.name)[0]
        else:
            # Prefer a copy file if present
            for c in candidates:
                name_low = c.name.lower()
                if "project_proposal" in name_low and ("копия" in name_low or "copy" in name_low):
                    docx = c
                    break
            else:
                if candidates:
                    docx = candidates[0]
    if not docx.exists():
        raise FileNotFoundError(docx)
    out_txt = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("_proposal_extracted.txt")
    text = extract_docx_text(docx)
    out_txt.write_text(text, encoding="utf-8")
    print(text[:1200])
    print(f"\nWrote {out_txt}, lines={len(text.splitlines())}, chars={len(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

