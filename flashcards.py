import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

def parse_flashcards(raw_text):
    cards = []
    lines = raw_text.split("\n")
    question, answer = None, None

    for line in lines:
        if line.startswith("Q:"):
            question = line.replace("Q:", "").strip()
        elif line.startswith("A:"):
            answer = line.replace("A:", "").strip()
            if question and answer:
                cards.append((question, answer))
                question, answer = None, None

    return cards

def generate_flashcards_from_summary(summary_text):
    """
    Generate flashcards from summarized content.
    Much faster and safer than using full chunks.
    """

    if not summary_text:
        return "No summary available."

    prompt = f"""
Create 8 high-quality study flashcards from the summary below.

Rules:
- Focus on core ideas.
- Keep answers concise.
- Format strictly as:

Q: ...
A: ...

Summary:
{summary_text}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 300
                }
            },
            timeout=90
        )

        if response.status_code == 200:
            return response.json().get("response", "")

        return f"Model error: {response.status_code}"

    except requests.exceptions.Timeout:
        return "⚠️ Flashcard generation timed out."
    except Exception as e:
        return f"Error: {str(e)}"