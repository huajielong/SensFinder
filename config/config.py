import os

# -------------------------- 1. 路径配置（必改：替换成你的实际路径） --------------------------
# 原始文本文件路径（一行一个字段）

# 项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 原始文本文件目录路径（会递归处理目录下所有文件）
RAW_FILES_PATH = os.path.join(PROJECT_ROOT, "data/input_raw/")
# 预处理后分批次文件的保存路径
BATCH_SAVE_PATH = os.path.join(PROJECT_ROOT, "data/preprocessed_batches/")
# LLM分类结果保存路径
CLASSIFY_SAVE_PATH = os.path.join(PROJECT_ROOT, "data/classify_results/")
# 验证出的问题字段保存路径
PROBLEM_SAVE_PATH = os.path.join(PROJECT_ROOT, "data/verify_problems/")

# -------------------------- 2. 预处理参数 --------------------------
# 每批次行数（建议1000-2000，避免LLM上下文超量）
BATCH_SIZE = 100
# 过滤无效字段：长度小于2的字段会被删除（如单字母“A”）
MIN_FIELD_LENGTH = 2

# -------------------------- 3. LLM配置（必改：填你的API Key） --------------------------
# 选择当前使用的模型服务："OPENAI" 或 "DEEPSEEK" 或 "LOCAL"
CURRENT_LLM_SERVICE = "DEEPSEEK"  # 这里可以轻松切换模型服务

# LLM温度（0.1-0.3，越低结果越稳定，分类用0.1足够）
LLM_TEMPERATURE = 0.1

# OpenAI配置
OPENAI_API_KEY = "sk-你的OPENAI API Key"
OPENAI_MODEL = "gpt-4o-mini"

# DeepSeek配置
DEEPSEEK_API_KEY = "sk-cad80889abd343599faab8cdc69956e4"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Local LLM配置
LOCAL_LLM_URL = "http://192.168.5.85:8000/v1/chat/completions"  # 本地LLM服务端点URL
LOCAL_LLM_MODEL = "Qwen2.5-72B-Instruct"  # 使用的模型名称
LOCAL_LLM_MAX_TOKENS = BATCH_SIZE*8  # 单次响应的最大令牌数（根据需求调整，避免过长响应）

# -------------------------- 4. 验证参数 --------------------------
# 低置信度阈值（低于此值的字段需人工复核，建议80）
LOW_CONFIDENCE_THRESHOLD = 80
# 公司名关键词（用于规则冲突校验，可补充）
COMPANY_KEYWORDS = ["Co., Ltd.", "Corp", "Inc", "LLC", "Group", "Company"]