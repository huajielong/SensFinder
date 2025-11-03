# SensFinder - 敏感信息识别系统

## 项目概述
SensFinder是一个强大的敏感信息识别系统，基于LLM技术自动识别和分类各类敏感信息，包括人名、地名、公司名、组织名、产品技术名等。系统采用模块化设计，提供从数据预处理、LLM分类到结果验证的完整解决方案。

## 系统架构

### 整体架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │    │                 │
│  数据预处理模块  │───>│   LLM分类模块   │───>│   结果验证模块   │───>│    输出结果     │
│  (data_preprocess) │  │  (llm_classify)  │  │ (result_verify) │  │                 │
│                 │    │                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 主要模块
1. **数据预处理模块**：清洗原始数据，过滤无效字段，批量保存
2. **LLM分类模块**：调用LLM服务对预处理后的数据进行分类
3. **结果验证模块**：验证分类结果，识别问题字段
4. **配置管理**：集中管理系统参数和LLM配置

## 快速开始

### 环境要求
- Python 3.8+
- 相关依赖包（见requirements.txt）

### 安装步骤
1. 克隆项目
```bash
git clone <repository_url>
cd SensFinder
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
创建`.env`文件，包含以下内容：
```
# LLM服务配置
# 选择OPENAI、DEEPSEEK或LOCAL
LLM_SERVICE=DEEPSEEK

# DeepSeek配置
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# OpenAI配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# 本地LLM配置
LOCAL_LLM_URL=http://localhost:8000/v1/chat/completions
```

### 使用方法

#### 1. 准备输入数据
将需要分析的文本文件放入指定目录中（默认：`./data/raw`）

#### 2. 运行主程序
```bash
python script/sens_finder.py
```

#### 3. 查看结果
- 分类结果：`./data/classified_results.csv`
- 问题字段：`./data/problems/all_problems.csv`
- 日志文件：`./logs/sens_finder.log`

## 配置说明

### 主要配置项
配置文件位于 `config/config.py`，主要配置包括：

- **路径配置**：数据目录、日志目录、模型目录等
- **预处理参数**：批次大小、最小字段长度等
- **LLM配置**：服务类型、API密钥、模型参数等
- **验证参数**：低置信度阈值、公司关键词等
- **并发控制**：LLM请求并发数
- **错误处理**：重试次数、重试间隔等

## 项目结构

```
SensFinder/
├── config/              # 配置目录
│   ├── config.py        # 主要配置文件
│   └── prompt_template.txt # 提示词模板
├── data/                # 数据目录
│   ├── raw/             # 原始数据
│   ├── batch/           # 批处理数据
│   ├── classified/      # 分类结果
│   └── problems/        # 问题字段
├── logs/                # 日志目录
├── script/              # 脚本目录
│   ├── data_preprocess.py # 数据预处理
│   ├── llm_classify.py   # LLM分类
│   ├── result_verify.py  # 结果验证
│   ├── sens_finder.py    # 主程序
│   └── local_llm*.py     # 本地LLM客户端
├── test/                # 测试目录
├── requirements.txt     # 依赖列表
└── README.md            # 项目文档
```

## 测试

运行测试脚本：
```bash
python test/test_sens_finder.py
```

## 支持的信息类型

1. **人名**：自然人姓名
2. **地名**：地理区域名称
3. **公司名及简称**：企业名称
4. **组织名及简称**：机构名称
5. **产品/技术名**：产品或技术名称
6. **邮箱地址**：电子邮件格式
7. **电话号码**：电话号码格式
8. **日期/时间**：日期时间信息

## 错误处理与日志

系统采用多级日志记录，包括：
- 文件日志：保存到`./logs`目录
- 控制台日志：实时显示运行状态

错误处理机制包括：
- 自动重试（指数退避）
- 异常捕获与记录
- 详细的错误信息输出

## 性能优化

- 批量处理：减少API调用次数
- 并发控制：合理利用资源
- 缓存机制：避免重复处理

## 注意事项

1. API密钥安全：请妥善保管API密钥，不要提交到代码仓库
2. 输入数据：确保输入数据编码为UTF-8
3. 性能调优：根据实际情况调整并发数和批次大小

## License

[MIT License](LICENSE)

## 项目简介

SensFinder是一款基于大语言模型(LLM)的自动化敏感信息识别与分类工具。它能够帮助用户快速发现和分类文本中的人名、地名、公司名和组织名等敏感信息，适用于数据脱敏前检查、数据泄露风险评估、合规性检查等场景。

## 功能特点

- **自动化识别**：使用LLM自动识别和分类敏感信息
- **多LLM支持**：支持OpenAI、DeepSeek和自定义本地LLM服务
- **灵活配置**：提供丰富的配置选项，适应不同需求
- **并行处理**：采用多线程并行处理，提高处理效率
- **结果验证**：自动检测低置信度结果和规则冲突
- **清晰报告**：生成结构化的问题字段报告，便于人工复核

## 系统架构

项目采用模块化设计，包含四个主要组件：

1. **主控制模块**：协调整个处理流程
2. **数据预处理模块**：清洗和分批次处理原始数据
3. **LLM分类模块**：调用LLM服务进行字段分类
4. **结果验证模块**：验证分类结果并生成问题报告

## 安装说明

### 环境要求

- Python 3.8 或更高版本
- 所需Python库：pandas, openai, requests

### 安装依赖

#### 创建虚拟环境并安装依赖

```bash
# 检查并创建虚拟环境
python -m venv venv

# 激活虚拟环境并安装依赖
venv\Scripts\Activate; pip install -r requirements.txt
```

#### 直接安装依赖（不使用虚拟环境）

```bash
pip install pandas openai requests
```

## 配置说明

在使用前，需要修改`config/config.py`文件中的配置项：

### 1. 路径配置

```python
# 原始文本文件目录路径（会递归处理目录下所有文件）
RAW_FILES_PATH = os.path.join(PROJECT_ROOT, "data/input_raw/")
# 预处理后分批次文件的保存路径
BATCH_SAVE_PATH = os.path.join(PROJECT_ROOT, "data/preprocessed_batches/")
# LLM分类结果保存路径
CLASSIFY_SAVE_PATH = os.path.join(PROJECT_ROOT, "data/classify_results/")
# 验证出的问题字段保存路径
PROBLEM_SAVE_PATH = os.path.join(PROJECT_ROOT, "data/verify_problems/")
```

### 2. 预处理参数

```python
# 每批次行数（建议1000-2000，避免LLM上下文超量）
BATCH_SIZE = 1000
# 过滤无效字段：长度小于2的字段会被删除
MIN_FIELD_LENGTH = 2
```

### 3. LLM配置

```python
# 选择当前使用的模型服务："OPENAI" 或 "DEEPSEEK" 或 "LOCAL"
CURRENT_LLM_SERVICE = "DEEPSEEK"

# LLM温度（0.1-0.3，越低结果越稳定）
LLM_TEMPERATURE = 0.1

# 根据选择的服务，配置相应的API密钥和模型信息
# OpenAI配置
OPENAI_API_KEY = "sk-你的OPENAI API Key"
OPENAI_MODEL = "gpt-4o-mini"

# DeepSeek配置
DEEPSEEK_API_KEY = "sk-你的DeepSeek API Key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Local LLM配置
LOCAL_LLM_URL = "http://localhost:8000/v1/chat/completions"
LOCAL_LLM_MODEL = "你的本地模型名称"
```

### 4. 验证参数

```python
# 低置信度阈值（低于此值的字段需人工复核，建议80）
LOW_CONFIDENCE_THRESHOLD = 80
# 公司名关键词（用于规则冲突校验）
COMPANY_KEYWORDS = ["Co., Ltd.", "Corp", "Inc", "LLC", "Group", "Company"]
```

## 使用方法

### 1. 准备输入数据

将需要分析的文本字段放入`data/input_raw/`目录下的文本文件中。支持以下格式：
- 每行一个字段
- 每行多个字段，使用空格分隔

### 2. 运行完整流程

执行主脚本，启动完整的处理流程：

```bash
python script/sens_finder.py
```

主脚本会按顺序执行以下操作：
1. 数据预处理：清洗、去重、分批次
2. LLM分类：调用配置的LLM服务进行分类
3. 结果验证：检测问题字段并生成报告

### 3. 单独运行各模块

也可以单独运行各个模块进行调试或特定操作：

- 数据预处理：
  ```bash
  python script/data_preprocess.py
  ```

- LLM分类：
  ```bash
  python script/llm_classify.py
  ```

- 结果验证：
  ```bash
  python script/result_verify.py
  ```

## 输出结果

执行完成后，会在以下路径生成相应的结果文件：

1. **预处理结果**：`data/preprocessed_batches/`
   - 分批次的CSV文件，如`batch_1.csv`, `batch_2.csv`等

2. **分类结果**：`data/classify_results/`
   - 对应批次的分类结果，如`result_batch_1.csv`, `result_batch_2.csv`等
   - 每个文件包含字段内容、分类结果、置信度和判断依据

3. **问题报告**：`data/verify_problems/all_problems.csv`
   - 汇总所有检测到的问题字段
   - 包含源批次、原始文本、分类、置信度、判断依据和问题类型

## 分类标准

系统将字段分为以下几类：

1. **人名**：自然人的完整姓名或常用名（如"John Smith"）
2. **地名**：地理区域名称（如"London"、"California State"）
3. **公司名及简称**：企业、商业机构的全称或简称（如"Apple Inc"）
4. **组织名及简称**：非盈利组织、政府机构等（如"UN"、"WHO"）
5. **未分类**：无法确定类别的字段

## 常见问题与解决方案

### 1. API调用失败

- 检查API密钥是否正确配置
- 确认网络连接是否正常
- 验证LLM服务是否可访问
- 查看日志中的具体错误信息

### 2. 分类准确率不高

- 调整LLM温度参数（降低到0.1左右）
- 使用更高级的LLM模型
- 更新提示词模板，提供更明确的分类规则

### 3. 处理速度慢

- 增加批次大小（但不要超过LLM上下文限制）
- 优化线程池配置，增加线程数
- 确保LLM服务响应速度快

### 4. 内存占用过高

- 减小批次大小
- 降低并行线程数
- 确保系统有足够的内存资源

## 扩展开发

### 添加新的分类类型

1. 修改`config/prompt_template.txt`文件，添加新的分类定义和示例
2. 如有需要，在`result_verify.py`中添加相应的验证规则

### 添加新的LLM服务

1. 在`config/config.py`中添加新服务的配置项
2. 在`llm_classify.py`中添加对应的API调用逻辑

### 添加新的验证规则

1. 在`result_verify.py`的`verify_results()`函数中添加新的验证逻辑
2. 定义相应的问题类型和筛选条件
3. 将新的问题字段添加到汇总结果中

## 注意事项

1. 确保API密钥的安全性，不要将包含密钥的配置文件提交到版本控制系统
2. 处理大量数据时，注意监控系统资源使用情况
3. 定期更新公司关键词列表，以提高规则冲突检测的准确性
4. 对于低置信度的分类结果，建议进行人工复核

## 许可证

[MIT License](LICENSE)