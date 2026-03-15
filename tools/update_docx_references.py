import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


@dataclass(frozen=True)
class Reference:
    idx: int
    text: str


REFERENCES: list[Reference] = [
    Reference(
        1,
        "Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. Analytical Chemistry, 36(8), 1627–1639.",
    ),
    Reference(
        2,
        "Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261–272.",
    ),
    Reference(
        3,
        "SciPy documentation (2026). scipy.signal.find_peaks — Peak finding. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html",
    ),
    Reference(
        4,
        "Ryan, C. G., Clayton, E., Griffin, W. L., Sie, S. H., & Cousens, D. R. (1988). SNIP: A statistics-sensitive background treatment for the quantitative analysis of PIXE spectra. Nuclear Instruments and Methods in Physics Research B, 34(3), 396–402.",
    ),
    Reference(
        5,
        "Thompson, P., Cox, D. E., & Hastings, J. B. (1987). Rietveld refinement of Debye–Scherrer synchrotron X-ray data from Al2O3. Journal of Applied Crystallography, 20(2), 79–83.",
    ),
    Reference(
        6,
        "Olivero, J. J., & Longbothum, R. L. (1977). Empirical fits to the Voigt line width: A brief review. Journal of Quantitative Spectroscopy and Radiative Transfer, 17(2), 233–236.",
    ),
    Reference(
        7,
        "Egami, T., & Billinge, S. J. L. (2012). Underneath the Bragg Peaks: Structural Analysis of Complex Materials (2nd ed.). Pergamon.",
    ),
    Reference(
        8,
        "Thakral, S., & Terban, M. W. (2021). Recent advances in the characterization of amorphous pharmaceuticals by X-ray diffractometry. Advanced Drug Delivery Reviews, 172, 1–25.",
    ),
    Reference(
        9,
        "Manoj, B. (2013). Study of stacking structure of amorphous carbon by X-ray diffraction technique. International Journal of Electrochemical Science, 8, 312–320.",
    ),
    Reference(
        10,
        "Cullity, B. D., & Stock, S. R. (2001). Elements of X-Ray Diffraction (3rd ed.). Prentice Hall.",
    ),
    Reference(
        11,
        "Moré, J. J., Garbow, B. S., & Hillstrom, K. E. (1980). User Guide for MINPACK-1. Argonne National Laboratory Report ANL-80-74.",
    ),
    Reference(
        12,
        "marimo documentation (2026). Reactive notebooks for Python. https://marimo.io/",
    ),
]


def _p_text(p: ET.Element) -> str:
    parts = [t.text for t in p.findall(".//w:t", NS) if t.text]
    return "".join(parts)


def _mk_run(text: str) -> ET.Element:
    r = ET.Element(f"{{{W_NS}}}r")
    t = ET.SubElement(r, f"{{{W_NS}}}t")
    # Preserve spaces if needed
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def _mk_para(text: str) -> ET.Element:
    p = ET.Element(f"{{{W_NS}}}p")
    p.append(_mk_run(text))
    return p


def _append_citation(p: ET.Element, citation: str) -> None:
    # Append as a separate run to avoid breaking existing run structure.
    p.append(_mk_run(citation))


def _find_paragraph_index(paras: list[ET.Element], predicate) -> int | None:
    for i, p in enumerate(paras):
        if predicate(_p_text(p).strip()):
            return i
    return None


def update_docx(in_path: Path, out_path: Path) -> None:
    with zipfile.ZipFile(in_path) as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    root = ET.fromstring(files["word/document.xml"])
    body = root.find(".//w:body", NS)
    if body is None:
        raise RuntimeError("word/document.xml: <w:body> not found")

    paras = list(body.findall("w:p", NS))

    # Insert two short paragraphs with citations so every reference is cited at least once.
    def is_methodology_heading(s: str) -> bool:
        s0 = s.strip().lower()
        return s0 in {"methodology", "methods"}

    meth_i = _find_paragraph_index(paras, is_methodology_heading)
    if meth_i is not None:
        p1 = _mk_para(
            "In this project, baseline estimation is performed using an iterative peak-clipping approach (SNIP)[4], "
            "the signal is smoothed using Savitzky–Golay filtering[1], peaks are detected via SciPy’s peak-finding routine[3], "
            "and peak shapes are modeled using pseudo-Voigt profiles commonly used in diffraction analysis[5][6]. "
            "Background and interpretation for amorphous diffraction data can be found in standard texts and reviews[7][8][9][10]."
        )
        p2 = _mk_para(
            "Implementation relies on SciPy for scientific computing and nonlinear least squares optimization (MINPACK)[2][11] "
            "and is packaged as a reactive marimo application for interactive analysis[12]."
        )
        body.insert(meth_i + 1, p1)
        body.insert(meth_i + 2, p2)

    # Refresh paragraph list after insertion
    paras = list(body.findall("w:p", NS))

    # Replace References section with numbered [n] list.
    def is_references_heading(s: str) -> bool:
        s0 = s.strip().lower()
        return s0 == "references" or s0.startswith("references")

    ref_i = _find_paragraph_index(paras, is_references_heading)
    if ref_i is None:
        # If not found, append at the end before any trailing signature/word count.
        ref_i = len(paras)
        body.append(_mk_para("References"))
        paras = list(body.findall("w:p", NS))
        ref_i = _find_paragraph_index(paras, is_references_heading) or ref_i

    # Find end of references: stop at a paragraph that looks like word count / author signature.
    end_i = len(paras)
    for j in range(ref_i + 1, len(paras)):
        s = _p_text(paras[j]).strip()
        s_low = s.lower()
        if s_low.startswith("words count") or s_low.startswith("word count") or s_low.startswith("shishlyannikov"):
            end_i = j
            break

    # Delete existing reference entries between ref_i+1 and end_i-1
    for p in paras[ref_i + 1 : end_i]:
        body.remove(p)

    # Insert new references right after heading
    insert_at = ref_i + 1
    for r in REFERENCES:
        body.insert(insert_at, _mk_para(f"[{r.idx}] {r.text}"))
        insert_at += 1

    new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    files["word/document.xml"] = new_xml

    with zipfile.ZipFile(out_path, "w") as zout:
        for name, data in files.items():
            zout.writestr(name, data)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/update_docx_references.py <in.docx> [out.docx]")
        return 2
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        # Work around Windows/PowerShell Unicode quoting issues (e.g., long dash in filename).
        candidates = list(Path(".").glob("*.docx"))
        for c in candidates:
            name_low = c.name.lower()
            if "project_proposal" in name_low and ("копия" in name_low or "copy" in name_low):
                in_path = c
                break
        else:
            if candidates:
                in_path = candidates[0]
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else in_path.with_suffix(".updated.docx")
    update_docx(in_path, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

