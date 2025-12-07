# LLM Gradio WebUI (text + RAG).

A Gradio web interface for text generation (Gemma3 4B as a default model).

## Screenshots

![Chat Interface](chat_2.png)
![RAG Chat Interface](rag_chat.png)

## Requirements

- NVIDIA GPU (8GB+ VRAM)
- Python 3.11+
- CUDA 12.1+

## Installation

1. Clone the repository:
```
git clone https://github.com/vpakarinen2/llm-text-gradio-webui.git
cd llm-text-gradio-webui
```

2. Create/activate virtual environment:
```
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Install PyTorch with CUDA:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

4. Install dependencies:
```
pip install -r requirements.txt --no-deps
```

5. Create `.env` file:
```
EMBED_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
HF_TOKEN=<HF_token_here>
MAX_NEW_TOKENS=128
DEVICE=cuda

GRADIO_ANALYTICS_ENABLED=False
```

## Hugging Face Token

1. Log in to Hugging Face
2. Create an access token (Settings → Access Tokens).
3. Log in:
```
huggingface-cli login
```

## Usage

```
python -m app.server
```

## Author

Ville pakarinen (@vpakarinen2)
