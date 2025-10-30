import os
import pandas as pd
import fitz  # PyMuPDF
from docx import Document

def read_file_to_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    elif ext == ".pdf":
        return _extract_pdf(path)
    elif ext == ".docx":
        return _extract_docx(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def _extract_pdf(path: str) -> pd.DataFrame:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    full = "\n".join(texts)
    return pd.DataFrame({"content": [full]})

def _extract_docx(path: str) -> pd.DataFrame:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full = "\n".join(paragraphs)
    return pd.DataFrame({"content": [full]})