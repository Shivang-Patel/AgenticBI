# AgenticBI

> An AI-powered Business Intelligence dashboard that lets users ask natural language questions about their data and automatically generates visualizations, SQL queries, and data insights.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Vizro](https://img.shields.io/badge/Vizro-0.1%2B-orange)](https://vizro.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Plotly-Dash-green)](https://dash.plotly.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Features

- **🔍 Natural Language Queries** — Ask questions in plain English, get instant answers
- **📊 Auto-Generated Visualizations** — Charts, graphs, and plots created automatically
- **📝 SQL Query Generation** — See the generated SQL behind every visualization
- **📋 Data Preview** — Inspect raw data in an expandable drawer
- **💾 Export & Save** — Download charts as PNG/CSV or save to workspace
- **🎨 Dark Theme** — Professional dark UI with smooth animations
- **⚡ Real-time Processing** — Live progress indicators during query execution

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│              User Interface              │
│         (Vizro + Plotly Dash)           │
├─────────────────────────────────────────┤
│         Natural Language Layer          │
│      (LangChain + LangGraph + LLM)     │
├─────────────────────────────────────────┤
│         SQL Generation Engine           │
│      (Schema-aware query builder)       │
├─────────────────────────────────────────┤
│         Data & Vector Store             │
│    (PostgreSQL + ChromaDB + Pandas)    │
└─────────────────────────────────────────┘
```

**Tech Stack:**
- **Frontend:** [Vizro](https://vizro.readthedocs.io) + [Plotly Dash](https://dash.plotly.com)
- **Backend:** Python 3.10+, LangChain, LangGraph
- **Database:** PostgreSQL (data), ChromaDB (embeddings)
- **LLM:** OpenAI GPT / Ollama (configurable)
- **Vector Search:** Sentence Transformers + ChromaDB

---

## 📦 Setup

### Prerequisites
- Python 3.10 or higher
- PostgreSQL (optional, SQLite fallback available)
- OpenAI API key or local Ollama instance

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Shivang-Patel/AgenticBI.git
cd AgenticBI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys and database settings

# 5. Run the application
python app/main.py
```

The dashboard will be available at `http://localhost:8051`

---

## 📁 Directory Structure

```
AgenticBI/
├── app/
│   ├── main.py              # Main application entry point
│   ├── config.py            # Configuration management
│   ├── assets/
│   │   └── custom.css       # Custom styling
│   └── __init__.py
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   └── test_main.py         # Main app tests
├── data/                    # Sample datasets
├── docs/                    # Documentation
├── .env.example             # Environment template
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata
├── Makefile                 # Common tasks
└── README.md                # This file
```

---

## 📸 Screenshots

> *Screenshots will be added here showing the app in action*

| Home | Query | Results |
|------|-------|---------|
| ![Home](docs/screenshots/home.png) | ![Query](docs/screenshots/query.png) | ![Results](docs/screenshots/results.png) |

---

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Lint code
make lint

# Run app locally
make run
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Vizro](https://vizro.readthedocs.io) by McKinsey
- Visualization powered by [Plotly](https://plotly.com)
- LLM orchestration via [LangChain](https://langchain.com)

---

**Made with ❤️ by [Kushagra Kshatri](mailto:kushagrakshatri16@gmail.com)**
