<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="v1.0"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT"/>
  <img src="https://img.shields.io/badge/python-3.8+-orange" alt="Python 3.8+"/>
  <img src="https://img.shields.io/github/stars/huajielong/SensFinder?style=social" alt="Stars"/>
  <img src="https://img.shields.io/badge/LLM-OpenAI%20%7C%20DeepSeek%20%7C%20Local-brightgreen" alt="LLM Support"/>
</p>

<h1 align="center">🔍 SensFinder</h1>
<p align="center"><b>LLM-Powered Sensitive Information Detection & Classification System</b></p>
<p align="center">
  🛡️ Pre-masking Inspection · 📋 Data Leak Risk Assessment · ✅ Compliance Checking
</p>

<p align="center">
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#-system-architecture">🏗️ Architecture</a> •
  <a href="#-key-features">⚡ Features</a> •
  <a href="#-classification-standard">📊 Classification</a> •
  <a href="#-configuration">⚙️ Config</a> •
  <a href="#-faq">❓ FAQ</a>
</p>

---

## 🤔 Do You Really Know What Sensitive Data Is Hidden in Your Text?

Data masking and compliance checks are essential for every organization, but manually reviewing thousands of text fields is nearly impossible:

| Pain Points | SensFinder Solutions |
|:------------|:--------------------|
| ❓ Thousands of records — manual review is impractical | ✅ **Automated Detection** — LLM-powered batch processing, done in minutes |
| ❓ Names, places, companies mixed together, hard to classify | ✅ **Smart Classification** — 6 precise categories with custom extension support |
| ❓ Switching between LLM providers is a hassle | ✅ **Multi-Engine** — OpenAI / DeepSeek / Local models, one-click switch |
| ❓ Results need manual verification | ✅ **Confidence Scoring** — Low-confidence fields auto-flagged for focused review |
| ❓ Batch processing can miss anomalies | ✅ **Multi-Level Validation** — Rule conflict detection + low confidence filtering |

### 🔥 Use Cases

> **Pre-masking Data Inspection** → **Privacy Compliance Audit** → **Data Leak Risk Assessment** → **PII Discovery & Classification**

---

## 🚀 Quick Start

### Prerequisites

| Dependency | Version |
|:-----------|:-------:|
| Python | 3.8+ |
| pandas | — |
| openai | — |

### One-Click Install

```bash
# 1. Clone the repo
git clone https://github.com/huajielong/SensFinder.git
cd SensFinder

# 2. Create virtual environment & install deps
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

### Configure LLM Service

Create a `.env` file (or modify `config/config.py` directly):

```ini
# Service selection: OPENAI / DEEPSEEK / LOCAL
LLM_SERVICE=DEEPSEEK

# DeepSeek Config
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# OpenAI Config
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1

# Local LLM Config
LOCAL_LLM_URL=http://localhost:8000/v1/chat/completions
```

### Run

```bash
python script/sens_finder.py
```

**That's it.** The program automatically runs: data preprocessing → LLM classification → result verification → report generation.

---

## 🏗️ System Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│                     │    │                     │    │                     │
│   Data Preprocess   │───>│   LLM Classification │───>│   Result Verify     │
│   data_preprocess   │    │   llm_classify      │    │   result_verify     │
│                     │    │                     │    │                     │
│  • Clean invalid    │    │  • Call LLM API     │    │  • Confidence score  │
│  • Batch sharding   │    │  • Smart tagging    │    │  • Conflict detect   │
│  • Dedup optimize   │    │  • Multi-thread     │    │  • Problem summary   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │                     │
                         │      Output         │
                         │  • Results CSV      │
                         │  • Problem Report   │
                         │  • Detailed Log     │
                         └─────────────────────┘
```

### Module Responsibilities

| Module | File | Core Responsibility |
|:-------|:-----|:--------------------|
| **Main Control** | `script/sens_finder.py` | Orchestrate full pipeline (preprocess → classify → verify) |
| **Data Preprocess** | `script/data_preprocess.py` | Clean, deduplicate, batch shard |
| **LLM Classify** | `script/llm_classify.py` | Call LLM to classify each field |
| **Local LLM Client** | `script/local_llm_client.py` | Support private/local model access |
| **Result Verify** | `script/result_verify.py` | Confidence scoring + conflict detection |
| **Configuration** | `config/config.py` | Centralized system and LLM config |

---

## ⚡ Key Features

| Feature | Description |
|:--------|:------------|
| 🤖 **Multi-LLM** | OpenAI GPT, DeepSeek, local/private models, one-click switch |
| 📦 **Batch Processing** | Auto-sharding for large text corpora (1000-2000 rows per batch) |
| 🎯 **Smart Classification** | Accurately identifies names, places, companies, organizations, products |
| ✅ **Confidence Scoring** | Each result scored; low confidence auto-flagged |
| 🔍 **Result Validation** | Rule conflict detection + low confidence filtering |
| ⚡ **Parallel Processing** | Multi-threaded for high throughput |
| 📊 **Structured Output** | CSV format for easy review and integration |

---

## 📊 Classification Standard

| Category | Examples |
|:---------|:---------|
| 👤 **Person Name** | John Smith, Zhang San, Li Si |
| 🌍 **Place Name** | London, California State, Beijing |
| 🏢 **Company Name** | Apple Inc, Alibaba, Tencent |
| 🏛️ **Organization** | UN, WHO, World Health Organization |
| 🔧 **Product/Tech** | iPhone 15, GPT-4, TensorFlow |
| 📧 **Other PII** | Email, phone number, date/time |

---

## ⚙️ Configuration

Config file: [`config/config.py`](config/config.py)

### LLM Config

```python
CURRENT_LLM_SERVICE = "DEEPSEEK"  # OPENAI | DEEPSEEK | LOCAL
LLM_TEMPERATURE = 0.1             # Lower = more stable (0.1-0.3)
```

### Preprocessing

```python
BATCH_SIZE = 1000           # Rows per batch
MIN_FIELD_LENGTH = 2        # Minimum field length filter
```

### Validation

```python
LOW_CONFIDENCE_THRESHOLD = 80    # Below this → manual review
```

---

## 📁 Project Structure

```
SensFinder/
├── config/                  # Configuration
│   ├── config.py            # Main config
│   └── prompt_template.txt  # LLM prompt template
├── data/                    # Data directory
│   ├── input_raw/           # Raw input
│   ├── preprocessed_batches/ # Preprocessed batches
│   ├── classify_results/    # Classification results
│   └── verify_problems/     # Verification issues
├── script/                  # Core scripts
│   ├── sens_finder.py       # Main entry point
│   ├── data_preprocess.py   # Data preprocessing
│   ├── llm_classify.py      # LLM classification
│   ├── local_llm_client.py  # Local LLM client
│   └── result_verify.py     # Result verification
├── test/                    # Tests
├── 产品设计PRD.md           # Product PRD (Chinese)
├── 技术实现方案.md          # Tech design doc (Chinese)
├── requirements.txt         # Python dependencies
└── README.md                # 💡 You are here
---

## ❓ FAQ

<details>
<summary><b>Which LLM providers are supported?</b></summary>
OpenAI (GPT-4o-mini/GPT-4o), DeepSeek (deepseek-chat), and any OpenAI-compatible local model (via Ollama/vLLM, etc.).
</details>

<details>
<summary><b>Processing is slow — what can I do?</b></summary>
1. Increase `BATCH_SIZE` (watch context limits)<br>
2. Check LLM API response speed<br>
3. Use a faster model (e.g., GPT-4o-mini)<br>
4. Check network stability
</details>

<details>
<summary><b>How to improve classification accuracy?</b></summary>
1. Lower `LLM_TEMPERATURE` to ~0.1<br>
2. Use a more capable LLM<br>
3. Add more examples to `prompt_template.txt`<br>
4. Lower `LOW_CONFIDENCE_THRESHOLD` for wider review scope
</details>

<details>
<summary><b>How to use the results?</b></summary>
Output is CSV with field content, classification, confidence score, and reasoning. Use for: pre-masking field marking, compliance audit evidence, data leak risk assessment reports.
</details>

<details>
<summary><b>Is my API key safe?</b></summary>
API keys are configured via `.env` or `config.py`. The project includes `.gitignore` to prevent accidental key commits.
</details>

---

## 🧪 Development & Extension

### Add New Classification Types
1. Edit `config/prompt_template.txt` — add definition and examples
2. Add corresponding validation rules in `result_verify.py`

### Add New LLM Service
1. Add config item in `config/config.py`
2. Add API call logic in `llm_classify.py`

### Testing
```bash
python test/test_sens_finder.py
```

---

## 🤝 Contributing

Contributions of all forms are welcome — issues, PRs, documentation improvements.

<a href="https://github.com/huajielong/SensFinder/graphs/contributors">
  <img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="Contributions Welcome"/>
</a>

## 📄 License

MIT © [huajielong](https://github.com/huajielong)

---

<p align="center">
  ⭐ Star SensFinder if it helps you protect your data!
</p>

> [中文说明](README.zh.md)
