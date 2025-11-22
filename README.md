# Universal Credit Act 2025 AI Analysis

This project is a Streamlit web app that leverages LangChain and Google Gemini / OpenAI Large Language Models (LLMs) to analyze the Universal Credit Act 2025 document. It performs multi-step AI tasks with the uploaded static PDF document, including:

- **Task 1:** Extracting clean, structured full text from the PDF.
- **Task 2:** Summarizing the Act into 5-10 bullet points summarizing key themes like purpose, definitions, eligibility, obligations, and enforcement.
- **Task 3:** Extracting key legislative sections into a structured JSON format.
- **Task 4:** Applying 6 legislative rule checks, outputting a JSON report with pass/fail status and evidence for each rule.

## Features

- Uses **FAISS vectorstore** built over document embeddings for efficient semantic retrieval.
- Supports **multiple LLM providers**: Google Gemini and OpenAI GPT-3.5, selectable in-app.
- Handles Google Gemini async calls with proper event loop management.
- Caches heavy computations (PDF processing, embeddings) for speed.
- Presents results and raw extracted text in an intuitive Streamlit UI.
- Saves a fully structured JSON report suitable for internship submission deliverables.

## Setup Instructions

1. **Clone this repository** and ensure `ukpga_20250022_en.pdf` is in the project root directory.

2. **Create a `.env` file** with your API keys:
    ```
    GOOGLE_API_KEY=your-google-gemini-api-key
    OPENAI_API_KEY=your-openai-api-key
    ```

3. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```

4. **Run the app**:
    ```
    streamlit run app.py
    ```

5. **Use the UI** to select your LLM model and click "Run Full Act Analysis".

## Project Structure

- `app.py`: Main Streamlit app with AI analysis pipeline.
- `ukpga_20250022_en.pdf`: Static PDF of Universal Credit Act 2025 to analyze.
- `universal_credit_act_2025_report.json`: Generated structured JSON report after analysis.
- `.env`: Environment variables with LLM API keys.
- `requirements.txt`: Python dependencies list.

## Design Considerations

- Extensive use of LangChain for document loading, embedding, retrieval, and chaining with LLMs.
- Careful async handling for Google Gemini API integration in Streamlit environment.
- Robust JSON extraction from LLM outputs cleaning markdown fences for validity.
- User-friendly and scalable design with caching.

## Deliverables

This repo enables generating:

- Clean full text extraction.
- Concise human-readable summary.
- Detailed JSON extraction of Act sections.
- Rule evaluations in JSON with evidence and confidence scores.
- Final comprehensive JSON report.

