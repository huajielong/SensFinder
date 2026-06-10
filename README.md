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

---

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="v1.0"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT"/>
  <img src="https://img.shields.io/badge/python-3.8+-orange" alt="Python 3.8+"/>
  <img src="https://img.shields.io/github/stars/huajielong/SensFinder?style=social" alt="Stars"/>
  <img src="https://img.shields.io/badge/LLM-OpenAI%20%7C%20DeepSeek%20%7C%20Local-brightgreen" alt="LLM Support"/>
</p>

<h1 align="center">🔍 SensFinder</h1>
<p align="center"><b>基于大语言模型的智能敏感信息识别与分类系统</b></p>
<p align="center">
  🛡️ 数据脱敏前检查 · 📋 数据泄露风险评估 · ✅ 合规性检查
</p>

<p align="center">
  <a href="#-快速开始">🚀 快速开始</a> •
  <a href="#-系统架构">🏗️ 系统架构</a> •
  <a href="#-核心功能">⚡ 核心功能</a> •
  <a href="#-分类标准">📊 分类标准</a> •
  <a href="#-配置指南">⚙️ 配置指南</a> •
  <a href="#-常见问题">❓ 常见问题</a>
</p>

---

## 🤔 你真的清楚数据里藏了多少敏感信息吗？

数据脱敏、合规检查是每个企业的必修课，但手动检查数万条文本字段几乎不可能：

| 你可能遇到的痛点 | SensFinder 帮你解决 |
|:-----------------|:--------------------|
| ❓ 数万行文本，人工逐条检查不现实 | ✅ **自动化识别** — LLM 驱动，批量处理，分钟级完成 |
| ❓ 人名、地名、公司名混杂，分类困难 | ✅ **智能分类** — 6 大类别精准区分，支持自定义扩展 |
| ❓ 不同 LLM 服务切换麻烦 | ✅ **多引擎支持** — OpenAI / DeepSeek / 本地模型一键切换 |
| ❓ 处理结果不可靠，需要人工复核 | ✅ **置信度评分** — 低置信度字段自动标记，复核有重点 |
| ❓ 批量处理容易漏掉异常 | ✅ **多级验证** — 规则冲突检测 + 低置信度筛选，不留死角 |

### 🔥 适用场景

> **数据脱敏前检查** → **隐私合规审计** → **数据泄露风险评估** → **PII 发现与分类**

---

## 🚀 快速开始

### 环境要求

| 依赖 | 版本 |
|:-----|:----:|
| Python | 3.8+ |
| pandas | — |
| openai | — |

### 一键安装

```bash
# 1. 克隆项目
git clone https://github.com/huajielong/SensFinder.git
cd SensFinder

# 2. 创建虚拟环境并安装依赖
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

### 配置 LLM 服务

创建 `.env` 文件（或直接修改 `config/config.py`）：

```ini
# 选择服务：OPENAI / DEEPSEEK / LOCAL
LLM_SERVICE=DEEPSEEK

# DeepSeek 配置
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# OpenAI 配置
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1

# 本地 LLM 配置
LOCAL_LLM_URL=http://localhost:8000/v1/chat/completions
```

### 运行

```bash
python script/sens_finder.py
```

**就是这么简单。** 程序会自动执行：数据预处理 → LLM 分类 → 结果验证 → 生成报告。

---

## 🏗️ 系统架构

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│                     │    │                     │    │                     │
│   数据预处理模块     │───>│    LLM 分类模块      │───>│    结果验证模块      │
│   data_preprocess   │    │   llm_classify      │    │   result_verify     │
│                     │    │                     │    │                     │
│  • 清洗无效字段      │    │  • 调用 LLM 服务     │    │  • 置信度评分       │
│  • 批量分片处理      │    │  • 智能分类打标      │    │  • 规则冲突检测     │
│  • 去重优化          │    │  • 多线程并发        │    │  • 问题字段汇总     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │                     │
                         │     输出结果         │
                         │  • 分类结果 CSV      │
                         │  • 问题字段报告      │
                         │  • 详细日志          │
                         └─────────────────────┘
```

### 模块职责

| 模块 | 文件 | 核心职责 |
|:-----|:-----|:---------|
| **主控制** | `script/sens_finder.py` | 协调完整处理流程（预处理 → 分类 → 验证） |
| **数据预处理** | `script/data_preprocess.py` | 清洗、去重、按批次分片 |
| **LLM 分类** | `script/llm_classify.py` | 调用 LLM 对每个字段分类打标 |
| **本地 LLM 客户端** | `script/local_llm_client.py` | 支持本地私有化模型接入 |
| **结果验证** | `script/result_verify.py` | 置信度评分 + 规则冲突检测 |
| **配置管理** | `config/config.py` | 集中管理系统参数和 LLM 配置 |

---

## ⚡ 核心功能

| 功能 | 说明 |
|:-----|:------|
| 🤖 **多 LLM 支持** | OpenAI GPT、DeepSeek、本地私有化模型，一键切换 |
| 📦 **批量处理** | 自动分片，支持大规模文本字段（单批次 1000-2000 行） |
| 🎯 **智能分类** | 精准识别人名、地名、公司名、组织名、产品技术名等 |
| ✅ **置信度评分** | 每个分类结果附带置信度，低分自动标记 |
| 🔍 **结果验证** | 规则冲突检测 + 低置信度筛选，确保可靠 |
| ⚡ **并行处理** | 多线程并发，大幅提升处理效率 |
| 📊 **结构化输出** | CSV 格式，方便后续人工复核与集成 |

---

## 📊 分类标准

系统将字段分为以下类别：

| 类别 | 示例 |
|:-----|:------|
| 👤 **人名** | John Smith、张三、李四 |
| 🌍 **地名** | London、California State、北京市 |
| 🏢 **公司名及简称** | Apple Inc、阿里巴巴、Tencent |
| 🏛️ **组织名及简称** | UN、WHO、世界卫生组织 |
| 🔧 **产品/技术名** | iPhone 15、GPT-4、TensorFlow |
| 📧 **其他 PII** | 邮箱地址、电话号码、日期时间 |

---

## ⚙️ 配置指南

配置文件位于 [`config/config.py`](config/config.py)，核心参数：

### LLM 配置

```python
# 选择模型服务
CURRENT_LLM_SERVICE = "DEEPSEEK"  # OPENAI | DEEPSEEK | LOCAL

# 模型温度（越低越稳定，建议 0.1-0.3）
LLM_TEMPERATURE = 0.1
```

### 预处理参数

```python
BATCH_SIZE = 1000           # 每批次行数
MIN_FIELD_LENGTH = 2        # 最小字段长度（过滤无效字段）
```

### 验证参数

```python
LOW_CONFIDENCE_THRESHOLD = 80    # 低置信度阈值（低于此值需人工复核）
```

---

## 📁 项目结构

```
SensFinder/
├── config/                  # 配置目录
│   ├── config.py            # 主要配置文件
│   └── prompt_template.txt  # LLM 提示词模板
├── data/                    # 数据目录
│   ├── input_raw/           # 原始输入数据
│   ├── preprocessed_batches/ # 预处理后的分片批次
│   ├── classify_results/    # LLM 分类结果
│   └── verify_problems/     # 验证出的问题字段
├── script/                  # 核心脚本
│   ├── sens_finder.py       # 主程序入口
│   ├── data_preprocess.py   # 数据预处理
│   ├── llm_classify.py      # LLM 分类
│   ├── local_llm_client.py  # 本地 LLM 客户端
│   └── result_verify.py     # 结果验证
├── test/                    # 测试
├── 产品设计PRD.md           # 产品需求文档
├── 技术实现方案.md          # 技术方案文档
├── requirements.txt         # Python 依赖
└── README.md                # 💡 你在这里
```

---

## ❓ 常见问题

<details>
<summary><b>支持哪些 LLM 服务？</b></summary>
目前支持 OpenAI（GPT-4o-mini/GPT-4o）、DeepSeek（deepseek-chat）以及任意兼容 OpenAI 接口格式的本地模型（如通过 Ollama/vLLM 部署的模型）。
</details>

<details>
<summary><b>处理速度慢怎么办？</b></summary>
1. 增大 `BATCH_SIZE`（注意不要超过 LLM 上下文限制）<br>
2. 确认 LLM 服务的响应速度<br>
3. 使用更快的模型（如 GPT-4o-mini 代替 GPT-4o）<br>
4. 检查网络连接是否稳定
</details>

<details>
<summary><b>如何提高分类准确率？</b></summary>
1. 降低 `LLM_TEMPERATURE` 到 0.1 左右<br>
2. 使用更强大的 LLM 模型<br>
3. 在 `prompt_template.txt` 中添加更多分类示例<br>
4. 适当降低 `LOW_CONFIDENCE_THRESHOLD` 扩大复核范围
</details>

<details>
<summary><b>分类结果怎么用？</b></summary>
输出为 CSV 格式，包含字段内容、分类结果、置信度和判断依据。可以直接用于：
- 数据脱敏前的敏感字段标记
- 合规审计的证据材料
- 数据泄露风险评估报告
</details>

<details>
<summary><b>API 密钥安全吗？</b></summary>
API 密钥通过 `.env` 文件或 `config.py` 配置，项目已内置 `.gitignore`，确保密钥不会误提交到代码仓库。
</details>

---

## 🧪 开发扩展

### 添加新分类类型

1. 修改 `config/prompt_template.txt`，添加新的分类定义和示例
2. 在 `result_verify.py` 中添加相应的验证规则

### 添加新 LLM 服务

1. 在 `config/config.py` 中添加新服务的配置项
2. 在 `llm_classify.py` 中添加对应的 API 调用逻辑

### 添加新验证规则

1. 在 `result_verify.py` 的 `verify_results()` 中新增验证逻辑
2. 定义对应的问题类型和筛选条件

---

## 🧪 测试

```bash
python test/test_sens_finder.py
```

---

## 🤝 贡献

欢迎任何形式的贡献——提交 Issue、Pull Request、改进文档或新增功能。

查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

<a href="https://github.com/huajielong/SensFinder/graphs/contributors">
  <img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="Contributions Welcome"/>
</a>

## 📄 License

MIT © [huajielong](https://github.com/huajielong)

---

<p align="center">
  ⭐ 如果 SensFinder 对你有帮助，请点个 Star 支持一下！
</p>
