# Visual Agentic Memory (VAM)

A high-performance framework for indexing long-horizon video streams and performing agentic retrieval over visual memory. VAM transforms raw video data into a searchable, structured "memory" that an AI agent can reason about to answer complex questions.

---

## 🌟 Core Capabilities

- **Visual Evidence**: Multi-tier image embeddings (using Gemini or Qwen3-VL) for precise visual recall.
- **Event Memory**: Automatically identifies and generates long-form event documents from video segments.
- **Summary Memory**: Creates reusable summary documents over custom time ranges and granularities.
- **Agentic Runtime**: A planning-first agent that orchestrates `retrieve` and `summarize` tools to solve multi-step reasoning tasks.
- **Persistent Storage**: High-performance SQLite backend for all memory types, ensuring your indexed data persists across sessions.
- **Unified Interface**: Modern terminal TUI for humans and a robust FastAPI/WebSocket API for machines.

---

## 🏗️ Architecture

VAM is built as a lean, modular Python package:

- **`vam/agent.py`**: The "brain" — handles task planning and tool orchestration.
- **`vam/retrieval/`**: The "memory engine" — manages indexing, vector search, and SQLite persistence.
- **`vam/vision/`**: The "eyes" — provides unified interfaces for image embedding backends.
- **`vam/server/`**: The "gateway" — FastAPI and WebSocket entry points for remote interaction.
- **`vam/models.py`**: The "schema" — unified data structures for the entire system.
- **`vam/cli.py`**: The "entry points" — exposes simplified commands for the TUI and Server.

---

## 🚦 Prerequisites

- **Python 3.10+**
- **FFmpeg**: Essential for video processing and frame extraction.
  ```bash
  # macOS
  brew install ffmpeg
  # Ubuntu/Linux
  sudo apt update && sudo apt install -y ffmpeg
  # Windows
  # Download from https://ffmpeg.org/download.html or use `choco install ffmpeg`
  ```

---

## 🚀 Quick Start

### 1. Installation

One-click install as a local package to automatically handle all Python dependencies:

```bash
pip install .
```

### 2. Configuration

Copy the example environment file and set your API key:

```bash
cp .env.example .env
```

Edit `.env` and set:
- `OPENROUTER_API_KEY`: Your OpenRouter API key (supports Gemini, etc.).
- `LLM_MODEL`: (Optional) Default is `google/gemini-3-flash-preview`.
- `EMBEDDING_MODEL`: (Optional) Default is `google/gemini-embedding-2-preview`.

### 3. Usage

VAM provides two main ways to interact:

#### **A. Interactive TUI (Recommended)**
Perfect for quick indexing and manual QA.
```bash
vam-tui
```
*   **Menu Options**: Overview, Index Video, Ask Agent, Summarize Range, Browse Memory, etc.

#### **B. API Server**
For programmatic integration or building custom frontends.
```bash
vam-server
```
*   **Swagger Docs**: `http://localhost:8000/docs`
*   **WebSocket Agent**: Connect to `ws://localhost:8000/ws/agent` for real-time streaming agentic chat.

---

## 📝 Example Queries

Once you've indexed a video, you can ask the agent complex questions like:

- *"What was the first thing I did after I finished reading the book?"*
- *"Find a scene where I am sitting on the sofa but not watching TV."*
- *"How many times did I drink coffee in the video?"*
- *"Summarize the first 30 minutes of the video, minute by minute, focusing on my kitchen activities."*

---

## ⚙️ Advanced Configuration

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LLM_MODEL` | Main LLM used for planning and answering | `google/gemini-3-flash-preview` |
| `EMBEDDING_MODEL` | Model used for image/text embeddings | `google/gemini-embedding-2-preview` |
| `FRAME_STORE_PATH` | Path to the SQLite database | `data/frame_store.sqlite3` |
| `VIDEO_FPS` | Default FPS for frame extraction | `1.0` |
| `VISION_EMBEDDING_BACKEND` | Embedding backend (`openrouter_gemini` or `qwen3_vl`) | `openrouter_gemini` |

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
