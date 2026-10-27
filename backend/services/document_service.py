import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from config import Config

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Config.UPLOADS_DIR
        self.active_document_id: Optional[str] = None
        self.documents_metadata: Dict[str, Dict] = {}

    def extract_text_from_file(self, file_path: Path) -> str:
        """Extrae el contenido en texto plano de un archivo (TXT, MD, PDF, etc.)"""
        ext = file_path.suffix.lower()

        if ext in (".txt", ".md", ".json", ".csv", ".log"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error leyendo archivo de texto: {e}")
                return ""

        elif ext == ".pdf":
            # Extraer con pypdfium2 o pypdf si está disponible
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(file_path)
                pages_text = []
                for page in pdf:
                    text_page = page.get_textpage()
                    pages_text.append(text_page.get_text_range())
                return "\n\n".join(pages_text)
            except Exception as e1:
                logger.warning(f"Error con pypdfium2 ({e1}), intentando con pypdf...")
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
                    return text
                except Exception as e2:
                    logger.error(f"No se pudo extraer texto del PDF: {e2}")
                    return ""

        return ""

    def save_and_process_upload(self, file_name: str, content_bytes: bytes) -> Dict:
        """Guarda el archivo subido y extrae su texto para usarlo como contexto en Hyper"""
        clean_name = Path(file_name).name
        target_path = self.upload_dir / clean_name

        with open(target_path, "wb") as f:
            f.write(content_bytes)

        extracted_text = self.extract_text_from_file(target_path)

        doc_info = {
            "id": clean_name,
            "filename": clean_name,
            "path": str(target_path),
            "size_bytes": len(content_bytes),
            "text_length": len(extracted_text),
            "content": extracted_text,
            "preview": extracted_text[:300] + ("..." if len(extracted_text) > 300 else "")
        }

        self.documents_metadata[clean_name] = doc_info
        self.active_document_id = clean_name
        logger.info(f"Documento '{clean_name}' procesado con éxito ({len(extracted_text)} caracteres)")
        return doc_info

    def list_documents(self) -> List[Dict]:
        """Devuelve la lista de archivos disponibles sin volcar el texto completo"""
        result = []
        for doc_id, data in self.documents_metadata.items():
            result.append({
                "id": doc_id,
                "filename": data["filename"],
                "size_bytes": data["size_bytes"],
                "text_length": data["text_length"],
                "preview": data["preview"],
                "is_active": (doc_id == self.active_document_id)
            })
        return result

    def get_active_document(self) -> Optional[Dict]:
        if self.active_document_id and self.active_document_id in self.documents_metadata:
            return self.documents_metadata[self.active_document_id]
        return None

    def set_active_document(self, doc_id: str) -> bool:
        if doc_id in self.documents_metadata:
            self.active_document_id = doc_id
            return True
        return False
