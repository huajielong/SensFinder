import os
import time
import logging
from datetime import datetime
import traceback

# 添加项目根目录到Python路径，确保能正确导入config等模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置模块
from config.config import (
    PROJECT_ROOT,
    BATCH_SAVE_PATH,
    CLASSIFY_SAVE_PATH,
    PROBLEM_SAVE_PATH
)

# 配置日志前确保logs目录存在
logs_dir = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'sens_finder.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_separator(title):
    """打印分隔线，用于区分不同阶段，并添加时间戳"""
    separator = f"\n{'-' * 60}"
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {title}"
    print(separator)
    print(timestamp)
    print(separator)
    logger.info(f"阶段开始: {title}")

def run_script(module_name, function_name):
    """运行指定模块中的指定函数，包含异常捕获和详细日志记录"""
    start_time = time.time()
    print(f"开始执行: {module_name}")
    logger.info(f"开始执行模块: {module_name}")
    
    try:
        # 使用动态导入方式
        module = __import__(module_name, fromlist=[function_name])
        # 执行函数
        getattr(module, function_name)()
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{module_name} 执行完成！耗时: {execution_time:.2f}秒")
        logger.info(f"模块 {module_name} 执行完成！耗时: {execution_time:.2f}秒")
        return True
    except Exception as e:
        error_msg = f"{module_name} 执行失败！错误: {type(e).__name__} - {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数，定义执行顺序并控制整个处理流程"""
    print_separator("开始敏感数据处理流程")
    
    # 定义要执行的模块和对应的主要函数
    scripts_to_run = [
        ("data_preprocess", "preprocess_data"),
        ("llm_classify", "batch_classify"),
        ("result_verify", "verify_results")
    ]
    
    # 按顺序执行每个模块
    for module_name, main_function in scripts_to_run:
        print_separator(f"执行 {module_name}")
        # 增加异常捕获，确保单个模块失败不会影响日志记录
        try:
            if not run_script(module_name, main_function):
                print_separator("敏感数据处理流程中断！")
                logger.critical("敏感数据处理流程因模块失败而中断")
                return
        except Exception as e:
            # 捕获run_script本身可能抛出的异常
            error_msg = f"执行模块 {module_name} 时发生未预期错误: {str(e)}"
            print(error_msg)
            logger.critical(error_msg)
            logger.error(traceback.format_exc())
            print_separator("敏感数据处理流程中断！")
            return
    
    print_separator("敏感数据处理流程全部完成！")
    logger.info("敏感数据处理流程全部完成")
    
    # 打印结果汇总
    print("\n处理结果汇总：")
    print(f"1. 数据预处理结果位于: {BATCH_SAVE_PATH}")
    print(f"2. LLM分类结果位于: {CLASSIFY_SAVE_PATH}")
    print(f"3. 验证问题字段位于: {PROBLEM_SAVE_PATH}")
    
    logger.info(f"处理结果汇总: 预处理结果位于 {BATCH_SAVE_PATH}, 分类结果位于 {CLASSIFY_SAVE_PATH}, 问题字段位于 {PROBLEM_SAVE_PATH}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断操作，处理流程已终止！")
        logger.warning("用户中断操作，处理流程已终止")
    except Exception as e:
        error_msg = f"\n发生未预期错误: {type(e).__name__} - {str(e)}"
        print(error_msg)
        logger.critical(error_msg)
        logger.error(traceback.format_exc())
    finally:
        # 确保日志文件正确关闭
        for handler in logger.handlers:
            handler.close()