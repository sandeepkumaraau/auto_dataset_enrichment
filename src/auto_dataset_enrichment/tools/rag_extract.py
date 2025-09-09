from crewai_tools import WebsiteSearchTool
import os
import google.generativeai as genai
google_api_key = os.getenv('GOOGLE_API_KEY')



# With custom Gemini configuration
rag_extract = WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="google",  # Use Google/Gemini
            config=dict(
                model='gemini/gemini-2.0-flash',
                api_key = google_api_key,
                temperature=0.4,
            ),
        ),
        embedder=dict(
            provider="google",  # Use Google embeddings
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
            ),
        ),
    )
)