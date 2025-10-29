import os
import time
from datetime import datetime

# 添加项目根目录到Python路径，确保能正确导入config等模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 由于已经创建了__init__.py文件，不再需要手动添加路径

def print_separator(title):
    """打印分隔线，用于区分不同阶段"""
    print(f"\n{'-' * 60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {title}")
    print(f"{'-' * 60}")

def run_script(module_name, function_name):
    """运行指定模块中的指定函数"""
    start_time = time.time()
    print(f"开始执行: {module_name}")
    
    try:
        # 使用直接导入方式，因为我们已经在script目录下
        module = __import__(module_name, fromlist=[function_name])
        # 执行函数
        getattr(module, function_name)()
        
        end_time = time.time()
        print(f"{module_name} 执行完成！耗时: {end_time - start_time:.2f}秒")
        return True
    except Exception as e:
        print(f"{module_name} 执行失败！错误: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数，按顺序执行三个处理脚本"""
    print_separator("开始敏感数据处理流程")
    
    # 定义要执行的模块和对应的主要函数（不再需要.py后缀）
    scripts_to_run = [
        ("data_preprocess", "preprocess_data"),
        ("llm_classify", "batch_classify"),
        ("result_verify", "verify_results")
    ]
    
    # 按顺序执行每个模块
    for module_name, main_function in scripts_to_run:
        print_separator(f"执行 {module_name}")
        if not run_script(module_name, main_function):
            print_separator("敏感数据处理流程中断！")
            return
    
    print_separator("敏感数据处理流程全部完成！")
    
    # 在主函数内部导入配置并打印结果汇总
    from config import BATCH_SAVE_PATH, CLASSIFY_SAVE_PATH, PROBLEM_SAVE_PATH
    print("\n处理结果汇总：")
    print(f"1. 数据预处理结果位于: {BATCH_SAVE_PATH}")
    print(f"2. LLM分类结果位于: {CLASSIFY_SAVE_PATH}")
    print(f"3. 验证问题字段位于: {PROBLEM_SAVE_PATH}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断操作，处理流程已终止！")
    except Exception as e:
        print(f"\n发生未预期错误: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()