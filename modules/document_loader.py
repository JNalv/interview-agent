"""Load and parse documents from a folder."""
from pathlib import Path
from typing import Tuple, List
from pypdf import PdfReader
from docx import Document
from config import SUPPORTED_EXTENSIONS

def load_documents(folder_path: str | Path) -> Tuple[str, int, List[str]]:
    """
    Load all supported documents from a folder.
    
    Args:
        folder_path: Path to folder containing documents
        
    Returns:
        Tuple of (concatenated_text, file_count, error_messages)
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return "", 0, [f"Path does not exist or is not a directory: {folder_path}"]
    
    texts = []
    errors = []
    file_count = 0
    
    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue
            
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        
        try:
            if ext in {".txt", ".md"}:
                text = _load_txt(file_path)
            elif ext == ".pdf":
                text = _load_pdf(file_path)
            elif ext == ".docx":
                text = _load_docx(file_path)
            else:
                continue
                
            if text.strip():
                texts.append(text)
                file_count += 1
        except Exception as e:
            errors.append(f"Error loading {file_path.name}: {str(e)}")
    
    concatenated = "\n\n".join(texts)
    return concatenated, file_count, errors

def _load_txt(file_path: Path) -> str:
    """Load plain text or markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()

def _load_pdf(file_path: Path) -> str:
    """Load PDF using pypdf."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text())
    return "\n".join(pages)

def _load_docx(file_path: Path) -> str:
    """Load DOCX using python-docx."""
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs]
    return "\n".join(paragraphs)

