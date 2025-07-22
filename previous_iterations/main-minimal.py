from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from io import BytesIO
import os, re, unicodedata, json, csv
from datetime import datetime

# Core extraction libraries
from PyPDF2 import PdfReader
import fitz
from docx import Document
from pptx import Presentation
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
import pandas as pd
from tempfile import NamedTemporaryFile
import xlrd

app = FastAPI(title="Document Parser for RAG", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class CleaningOptions(BaseModel):
    normalize_unicode: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    clean_whitespace: bool = True
    preserve_structure: bool = True
    max_length: int = 100000

class DocumentResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    file_info: Dict[str, Any]
    processing_time: float

class RAGTextCleaner:
    def __init__(self, options: CleaningOptions):
        self.options = options
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        if self.options.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
        
        if self.options.remove_urls:
            text = re.sub(r'http[s]?://\S+', '', text)
        
        if self.options.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        if self.options.clean_whitespace:
            if self.options.preserve_structure:
                text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                text = re.sub(r' *\n *', '\n', text)
            else:
                text = re.sub(r'\s+', ' ', text)
        
        if len(text) > self.options.max_length:
            text = text[:self.options.max_length].rsplit(' ', 1)[0] + "..."
        
        return text.strip()

def extract_pdf_text(file_bytes: bytes) -> tuple[str, dict]:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        metadata = {"pages": doc.page_count, "method": "PyMuPDF"}
        doc.close()
    except Exception:
        reader = PdfReader(BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        metadata = {"pages": len(reader.pages), "method": "PyPDF2"}
    return text.strip(), metadata

def extract_docx_text(file_bytes: bytes) -> tuple[str, dict]:
    doc = Document(BytesIO(file_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    
    table_text = []
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells]
            table_text.append(" | ".join(row_text))
    
    text = "\n".join(paragraphs)
    if table_text:
        text += "\n\n" + "\n".join(table_text)
    
    return text.strip(), {"paragraphs": len(paragraphs), "tables": len(doc.tables)}

def extract_pptx_text(file_bytes: bytes) -> tuple[str, dict]:
    prs = Presentation(BytesIO(file_bytes))
    slides_text = []
    
    for i, slide in enumerate(prs.slides, 1):
        slide_content = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_content.append(shape.text.strip())
        if slide_content:
            slides_text.append(f"Slide {i}:\n" + "\n".join(slide_content))
    
    return "\n\n".join(slides_text), {"slides": len(prs.slides)}

def extract_image_text(file_bytes: bytes) -> tuple[str, dict]:
    image = Image.open(BytesIO(file_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    text = pytesseract.image_to_string(image)
    return text.strip(), {"image_size": image.size}

def extract_csv_text(file_bytes: bytes) -> tuple[str, dict]:
    try:
        df = pd.read_csv(BytesIO(file_bytes))
        return df.to_string(index=False), {"rows": len(df), "columns": len(df.columns)}
    except Exception:
        text_content = file_bytes.decode("utf-8", errors="ignore")
        rows = list(csv.reader(text_content.splitlines()))
        return "\n".join([", ".join(row) for row in rows]), {"rows": len(rows)}

def extract_excel_text(file_bytes: bytes, ext: str) -> tuple[str, dict]:
    with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if ext == ".xlsx":
            df_dict = pd.read_excel(tmp_path, sheet_name=None)
            sheets_text = [f"Sheet: {name}\n{df.to_string(index=False)}" 
                          for name, df in df_dict.items()]
            return "\n\n".join(sheets_text), {"sheets": len(df_dict)}
        else:
            book = xlrd.open_workbook(tmp_path)
            sheets_text = []
            for sheet in book.sheets():
                sheet_data = []
                for row_idx in range(sheet.nrows):
                    row = [str(sheet.cell_value(row_idx, col)) for col in range(sheet.ncols)]
                    sheet_data.append(", ".join(row))
                sheets_text.append(f"Sheet: {sheet.name}\n" + "\n".join(sheet_data))
            return "\n\n".join(sheets_text), {"sheets": len(book.sheets())}
    finally:
        os.remove(tmp_path)

def extract_txt_text(file_bytes: bytes) -> tuple[str, dict]:
    return file_bytes.decode("utf-8", errors="ignore").strip(), {"encoding": "utf-8"}

def extract_html_text(file_bytes: bytes) -> tuple[str, dict]:
    soup = BeautifulSoup(file_bytes, "html.parser")
    text = soup.get_text(separator='\n').strip()
    return text, {"title": soup.title.string if soup.title else ""}

def extract_json_text(file_bytes: bytes) -> tuple[str, dict]:
    try:
        data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        return json.dumps(data, indent=2, ensure_ascii=False), {"valid": True}
    except json.JSONDecodeError as e:
        return "Invalid JSON file.", {"valid": False, "error": str(e)}

EXTRACTORS = {
    ".pdf": extract_pdf_text, ".docx": extract_docx_text, ".pptx": extract_pptx_text,
    ".png": extract_image_text, ".jpg": extract_image_text, ".jpeg": extract_image_text,
    ".txt": extract_txt_text, ".html": extract_html_text, ".csv": extract_csv_text,
    ".json": extract_json_text, ".xlsx": extract_excel_text, ".xls": extract_excel_text,
}

@app.get("/")
async def root():
    return {"message": "Document Parser for RAG", "supported_formats": list(EXTRACTORS.keys())}

@app.post("/parse", response_model=DocumentResult)
async def parse_file(
    file: UploadFile = File(...),
    normalize_unicode: bool = Query(True),
    remove_urls: bool = Query(True),
    remove_emails: bool = Query(True),
    clean_whitespace: bool = Query(True),
    preserve_structure: bool = Query(True),
    max_length: int = Query(100000)
):
    start_time = datetime.now()
    
    try:
        content = await file.read()
        ext = os.path.splitext(file.filename)[-1].lower()
        
        if ext not in EXTRACTORS:
            raise HTTPException(400, f"Unsupported file type: {ext}")
        
        # Extract text
        if ext in [".xlsx", ".xls"]:
            raw_text, metadata = EXTRACTORS[ext](content, ext)
        else:
            raw_text, metadata = EXTRACTORS[ext](content)
        
        # Clean text
        options = CleaningOptions(
            normalize_unicode=normalize_unicode,
            remove_urls=remove_urls,
            remove_emails=remove_emails,
            clean_whitespace=clean_whitespace,
            preserve_structure=preserve_structure,
            max_length=max_length
        )
        
        cleaner = RAGTextCleaner(options)
        cleaned_text = cleaner.clean_text(raw_text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DocumentResult(
            text=cleaned_text,
            metadata=metadata,
            processing_time=processing_time,
            file_info={
                "filename": file.filename,
                "file_size": len(content),
                "file_type": ext
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)