
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Gemini API key not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)


def generate_answer(question, context):
    """Generate an answer with retries for temporary API errors."""

    prompt = f"""
You are a careful PDF study assistant.

Answer the user's question using ONLY the information
provided in the PDF context.

Follow these rules strictly:
1. Answer the exact question asked.
2. Do not add unsupported facts.
3. If context is insufficient, clearly say:
   "The provided PDF context does not contain enough
   information to answer this fully."
4. Do not guess or invent facts.
5. Combine information from multiple passages when needed.
6. Explain clearly in simple language.
7. Preserve important technical terms and details.
8. Mention conflicting information if present.
9. Keep the answer focused.

PDF context:
{context}

User's question:
{question}

Answer based strictly on the PDF context:
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            if response.text:
                return response.text

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        except errors.APIError as error:
            status_code = getattr(error, "code", None)

            # Retry temporary server errors and rate limits
            if status_code in (429, 500, 502, 503, 504):
                if attempt < max_attempts - 1:
                    wait_seconds = 2 ** (attempt + 1)
                    print(
                        f"Temporary Gemini error "
                        f"({status_code}). Retrying in "
                        f"{wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)
                    continue

                raise RuntimeError(
                    "Gemini is temporarily busy or rate-limited. "
                    "Please try again in a little while."
                ) from error

            # Do not retry permanent API errors
            raise RuntimeError(
                f"Gemini API error ({status_code}). "
                "Check your API key, model access, and request."
            ) from error

        except (TimeoutError, ConnectionError) as error:
            if attempt < max_attempts - 1:
                time.sleep(2 ** (attempt + 1))
                continue

            raise RuntimeError(
                "Could not connect to Gemini after 3 attempts. "
                "Check your internet connection and try again."
            ) from error


if __name__ == "__main__":
    sample_context = """
    Gradient descent is an optimization algorithm.
    It updates a model's parameters to reduce its loss.
    """

    sample_question = "What is gradient descent?"

    answer = generate_answer(sample_question, sample_context)

    print("\nAI Answer:")
    print(answer)
