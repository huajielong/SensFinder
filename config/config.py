import os
# 从.env文件加载环境变量（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("警告: python-dotenv 模块未安装，环境变量将从系统读取")

# 加载环境变量的辅助函数
def get_env_variable(var_name, default=None, required=False):
    """
    获取环境变量，如果不存在且required=True则抛出异常，否则返回默认值
    
    Args:
        var_name: 环境变量名称
        default: 默认值
        required: 是否必需
        
    Returns:
        环境变量值或默认值
        
    Raises:
        ValueError: 当环境变量不存在且required=True时
    """
    value = os.environ.get(var_name, default)
    if required and value is None:
        raise ValueError(f"必需的环境变量 {var_name} 未设置")
    return value

# -------------------------- 1. 路径配置 --------------------------
# 项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 原始文本文件目录路径（会递归处理目录下所有文件）
RAW_FILES_PATH = os.path.join(PROJECT_ROOT, "data/input_raw/")
# 预处理后分批次文件的保存路径
BATCH_SAVE_PATH = os.path.join(PROJECT_ROOT, "data/preprocessed_batches/")
# LLM分类结果保存路径
CLASSIFY_SAVE_PATH = os.path.join(PROJECT_ROOT, "data/classification_results/")
# 验证出的问题字段保存路径
PROBLEM_SAVE_PATH = os.path.join(PROJECT_ROOT, "data/problematic_fields/")
# 提示词模板文件路径
PROMPT_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "config/prompt_template.txt")

# -------------------------- 2. 预处理参数 --------------------------
# 每批次行数（建议1000-2000，避免LLM上下文超量）
BATCH_SIZE = 1000
# 过滤无效字段：长度小于2的字段会被删除（如单字母"A"）
MIN_FIELD_LENGTH = 2

# -------------------------- 3. LLM配置 --------------------------
# 选择当前使用的模型服务："OPENAI" 或 "DEEPSEEK" 或 "LOCAL"
LLM_SERVICE = "DEEPSEEK"  # 这里可以轻松切换模型服务

# LLM温度（0.1-0.3，越低结果越稳定，分类用0.1足够）
TEMPERATURE = 0.1

# OpenAI配置
# 使用辅助函数获取环境变量，提供默认值
OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
OPENAI_MODEL = get_env_variable("OPENAI_MODEL")  # 默认使用gpt-4o模型

# DeepSeek配置
# 使用辅助函数获取环境变量，提供默认值
DEEPSEEK_API_KEY = get_env_variable("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = get_env_variable("DEEPSEEK_BASE_URL")  # 默认API基础URL
DEEPSEEK_MODEL = get_env_variable("DEEPSEEK_MODEL")  # 默认使用deepseek-chat模型

# Local LLM配置
LOCAL_LLM_URL = get_env_variable("LOCAL_LLM_URL")  # 默认本地URL
LOCAL_LLM_MODEL = get_env_variable("LOCAL_LLM_MODEL")  # 默认模型名称
LOCAL_LLM_MAX_TOKENS = BATCH_SIZE*8  # 单次响应的最大令牌数

# 并发配置
# 根据不同的LLM服务提供商设置不同的并发数
LLM_CONCURRENCY = {
    "OPENAI": 5,
    "DEEPSEEK": 5,
    "LOCAL": 3  # 本地LLM并发数较低，考虑本地资源限制
}

# -------------------------- 4. 验证参数 --------------------------
# 低置信度阈值（低于此值的字段需人工复核，建议80）
LOW_CONFIDENCE_THRESHOLD = 80
# 公司名关键词（用于规则冲突校验，可补充）
COMPANY_NAME_KEYWORDS = ["Co., Ltd.", "Corp", "Inc", "LLC", "Group", "Company", "Limited", "Ltd", "株式会社"]

# -------------------------- 5. 错误处理配置 --------------------------
# API调用最大重试次数
MAX_RETRY_COUNT = 3
# 初始重试间隔（秒）
INITIAL_RETRY_INTERVAL = 1.0
# 重试间隔乘数（用于指数退避策略）
RETRY_INTERVAL_MULTIPLIER = 2.0
# API请求超时时间（秒）
API_TIMEOUT = 600