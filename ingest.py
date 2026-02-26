import fitz
import pytesseract
from PIL import Image
import io
from typing import Union

def extract_text_from_pdf(file: Union[str, io.BytesIO]) -> str:
    """
    Extract text from PDF.
    Accepts either:
    - file path (str)
    - Streamlit uploaded file (BytesIO)
    """

    extracted_text = ""

    try:
        if isinstance(file, io.BytesIO):
            pdf = fitz.open(stream=file.read(), filetype="pdf")
        else:
            pdf = fitz.open(file)
    except Exception as e:
        raise ValueError(f"Failed to open PDF file: {e}")

    for page_number, page in enumerate(pdf, start=1):
        try:
            text = page.get_text("text")

            if text and len(text.strip()) > 50:
                extracted_text += text
            else:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes()))
                ocr_text = pytesseract.image_to_string(img)
                extracted_text += ocr_text

        except Exception as page_error:
            print(f"Warning: Skipping page {page_number}: {page_error}")

    pdf.close()

    if not extracted_text.strip():
        raise ValueError("No readable text could be extracted from the PDF.")

    return extracted_text