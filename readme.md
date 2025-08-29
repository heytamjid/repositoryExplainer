# Repository Explainer

The "Repository Explainer" is a webapp designed to provide AI-powered insights into GitHub repositories. Its primary purpose is to enhance the understanding and exploration of large code repositories through generative agentic AI system and retrieval-augmented generation (RAG) techniques.

Repository Explainer leverages a multi-layered RAG framework with an AI agent to address the limitations of large language models when processing entire codebases. It features an intelligent dual-granularity indexing system, both at file-level and at logical-unit level, preserving both the context and the precision. Logical units are extracted using tree-sitter based utilities to enable fine-grained sementic analysis of the codebase. The system can generate high-level summaries of a repository’s architecture, data flow & content and also answer developer queries in natural language by retrieving and integrating relevant code context. Additionally, an AI agent is integrated that can fetch specific files or functions on demand.

## Features

*   **Generate Documentation/Summaries:** Users can input a GitHub repository URL to receive a high-level explanation or summary (i.e. purpose & scope, system architecture, data flow, core business logic etc) of its contents. The application includes a caching mechanism for quick retrieval of previously generated summaries.
*   **Engage in Q&A:** Once a repository has been indexed, users can ask natural language questions (i.e. explain the dataflow, how user input is validated, where this functionality is implemented etc.) and the system can give context-aware answer by retriving the most relevant code snippets across the codebase and agentically fetching specific files or functions as needed. 


## Deployment
Currently deployed on DigitalOcean and can be accessed at:
[http://repoexplainer.tamjid.me/](http://repoexplainer.tamjid.me/)


## Running Locally

To run the Repository Explainer locally, follow these steps:

### Prerequisites

*   Python 3.8+
*   A GitHub Personal Access Token (for `GITHUB_TOKEN`)
*   A Google API Key (for `GOOGLE_API_KEY`) with access to Generative AI services.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd repositoryExplainer # Or the name of your cloned directory
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The project uses a `requirements.txt` file for its dependencies.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root of the `repositoryExplainer` directory (the same level as `manage.py`). Add your API keys and Django secret key:

    ```ini
    GITHUB_TOKEN='your_github_personal_access_token'
    GOOGLE_API_KEY='your_google_api_key'
    ```
    *   **`DJANGO_SECRET_KEY`**: Essential for Django's security.
    *   **`GITHUB_TOKEN`**: Required for fetching repository content from GitHub. You can get this from your GitHub account settings under Developer Settings -> Personal Access Tokens for free. 
    *   **`GOOGLE_API_KEY`**: Required for interacting with Google's Generative and Embedding models . You can obtain this simply from [Google AI Studio](https://aistudio.google.com/app/apikey) for free. Alternatively, you can obtain it from [Google Cloud Console](https://console.cloud.google.com/). You have to enable the Generative AI API Service in the later case.

### Using the local embedding model (optional)

If you prefer to run embeddings locally (as you will run out of the quota for the embedding models pretty fast on free tier), this project supports a local sentence-transformers based embedding path.

1. Set the embedding mode to local in your `.env`:

```ini
EMBEDDING_MODE='local'
LOCAL_EMBEDDING_MODEL='nomic-ai/nomic-embed-text-v1.5'
LOCAL_EMBEDDING_DEVICE='cpu'  # or 'cuda' if you have a supported GPU
# Optional tuning
EMBEDDING_BATCH_SIZE=100
```

* The default remote embedding mode is `remote`. To switch to local set `EMBEDDING_MODE=local` in your environment or pass `embedding_mode='local'` to the `index_repository(...)` or `query_repository(...)` functions in `myapp/embedder.py`.
* The code will lazily load the model specified by `LOCAL_EMBEDDING_MODEL`. Model weights are downloaded on first use and can be large — ensure you have enough disk space.
* For GPU use set `LOCAL_EMBEDDING_DEVICE='cuda'` (or an appropriate CUDA device string) and ensure PyTorch with CUDA is installed. If you see memory or OOM errors, reduce `EMBEDDING_BATCH_SIZE`.
* If you don't want to install local dependencies, keep `EMBEDDING_MODE` set to `remote` and provide `GOOGLE_API_KEY` instead.
* See `myapp/embedder.py` for the exact environment variable names, default model IDs, and the per-call `embedding_mode` override.

5.  **Apply Django Migrations:**
    ```bash
    python manage.py migrate
    ```

### Running the Application

1.  **Start the Django development server:**
    ```bash
    python manage.py runserver
    ```

2.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:8000/`.

    *   You will see the "Explain Repository" interface. Enter a GitHub repository URL to generate a summary.
    *   Click "ASK QUESTION" to navigate to the Q&A interface. Here, you'll first need to "Index Repository" for the Q&A feature to work.