import re
from typing import List

def clean_text(text: str) -> str:
    """
    Lowercase text and normalize whitespace.
    Keep line breaks and common code symbols.
    """
    text = text.lower()
    # Normalize spaces but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n", "\n", text)  # unify line endings
    # Keep letters, numbers, punctuation, and common code symbols
    text = re.sub(r"[^a-z0-9\s.,!?(){}\[\]:;<>+=\-*/#\"\']+", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> list[str]:
    lines = text.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 <= chunk_size:
            current_chunk += line + "\n"
        else:
            # Append current chunk
            chunks.append(current_chunk.strip())

            # Start new chunk with safe overlap
            if chunks:
                last_chunk_lines = chunks[-1].split("\n")
                overlap_lines = last_chunk_lines[-overlap:] if len(last_chunk_lines) >= overlap else last_chunk_lines
                current_chunk = "\n".join(overlap_lines) + "\n" + line
            else:
                current_chunk = line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
