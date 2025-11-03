import pandas as pd
import os
import concurrent.futures
import time
import random
import glob
import logging
import traceback
from datetime import datetime
import sys
from openai import OpenAI, APIError, Timeout

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从config.config正确导入配置
from config.config import (
    BATCH_SAVE_PATH,
    CLASSIFY_SAVE_PATH,
    LLM_SERVICE,
    TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    PROJECT_ROOT,
    LLM_CONCURRENCY,
    MAX_RETRY_COUNT,
    INITIAL_RETRY_INTERVAL,
    RETRY_INTERVAL_MULTIPLIER,
    PROMPT_TEMPLATE_PATH,
    API_TIMEOUT
)

# 导入本地LLM客户端
from script.local_llm_client import LocalLLMClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs/llm_classify.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def merge_classification_results():
    """
    合并分类结果文件
    
    功能：合并分类结果目录下的所有CSV文件到一个总表中，并按category字段排序
    结果保存到merged_results.csv文件
    """
    try:
        input_dir = CLASSIFY_SAVE_PATH
        output_file = os.path.join(PROJECT_ROOT, 'data', 'merged_results.csv')
        
        logger.info(f"开始合并分类结果文件...")
        print(f"\n开始合并分类结果文件...")
        
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
        
        if not csv_files:
            warning_msg = f"没有找到CSV文件在目录: {input_dir}"
            logger.warning(warning_msg)
            print(warning_msg)
            return
        
        logger.info(f"找到 {len(csv_files)} 个CSV文件进行合并...")
        print(f"找到 {len(csv_files)} 个CSV文件进行合并...")
        
        # 初始化一个空的DataFrame
        all_data = pd.DataFrame()
        
        # 读取并合并所有CSV文件
        for file in csv_files:
            try:
                # 读取单个CSV文件
                df = pd.read_csv(file)
                
                # 添加源文件信息
                df['source_file'] = os.path.basename(file)
                
                # 合并到总DataFrame
                all_data = pd.concat([all_data, df], ignore_index=True)
                logger.info(f"成功合并文件: {os.path.basename(file)}")
            except Exception as e:
                error_msg = f"处理文件 {file} 时出错: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
        
        # 检查是否成功合并了数据
        if all_data.empty:
            warning_msg = "警告: 没有成功合并任何数据!"
            logger.warning(warning_msg)
            print(warning_msg)
            return
        
        logger.info(f"合并完成! 总数据行数: {len(all_data)}")
        print(f"合并完成! 总数据行数: {len(all_data)}")
        
        # 按category字段排序
        if 'category' in all_data.columns:
            all_data = all_data.sort_values(by='category', ignore_index=True)
            logger.info("已按 category 字段排序")
            print("已按 category 字段排序")
        else:
            warning_msg = "警告: 数据中没有 'category' 字段，无法排序"
            logger.warning(warning_msg)
            print(warning_msg)
        
        # 保存合并后的数据
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            all_data.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"合并结果已保存到: {output_file}")
            print(f"合并结果已保存到: {output_file}")
            
            # 显示一些基本统计信息
            logger.info(f"数据统计信息 - 总列数: {len(all_data.columns)}, 总行数: {len(all_data)}")
            print(f"\n数据统计信息:")
            print(f"总列数: {len(all_data.columns)}")
            print(f"总行数: {len(all_data)}")
            
            if 'category' in all_data.columns:
                logger.info(f"唯一 category 数量: {len(all_data['category'].value_counts())}")
                print(f"\nCategory 分布:")
                category_counts = all_data['category'].value_counts()
                print(category_counts)
                print(f"\n唯一 category 数量: {len(category_counts)}")
                
        except Exception as e:
            error_msg = f"保存文件时出错: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
    except Exception as e:
        error_msg = f"合并分类结果时发生未预期错误: {str(e)}"
        logger.critical(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)

def load_prompt_template():
    """
    加载LLM提示词模板
    
    返回:
        str: 提示词模板内容
    """
    try:
        template_path = PROMPT_TEMPLATE_PATH
        logger.info(f"尝试加载提示词模板: {template_path}")
        
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        
        if not template.strip():
            raise ValueError("提示词模板内容为空")
            
        logger.info("提示词模板加载成功")
        return template
    except FileNotFoundError:
        error_msg = f"读取Prompt模板失败！文件不存在：{template_path}"
        logger.error(error_msg)
        print(error_msg)
        return None
    except Exception as e:
        error_msg = f"读取Prompt模板失败！错误：{e}"
        logger.error(error_msg)
        print(error_msg)
        return None

def classify_single_batch(batch_file_path, prompt_template):
    """
    调用LLM分类单个批次（带重试机制）
    
    参数:
        batch_file_path (str): 批次文件路径
        prompt_template (str): 提示词模板
    
    返回:
        pd.DataFrame: 分类结果数据框
    """
    try:
        logger.info(f"开始处理批次文件: {os.path.basename(batch_file_path)}")
        
        # 读取批次文件
        batch_df = pd.read_csv(batch_file_path, encoding="utf-8")
        fields = batch_df["raw_text"].tolist()
        
        logger.info(f"批次文件包含 {len(fields)} 个字段")
        
        # 格式化字段为"1. 字段1\n2. 字段2"格式
        fields_text = "\n".join([f"{i+1}. {field}" for i, field in enumerate(fields)])
        # 填充Prompt模板
        final_prompt = prompt_template.replace("{{fields_text}}", fields_text)
        
        logger.info("准备发送请求到LLM服务")
        
        # 执行带重试的LLM调用
        for retry in range(MAX_RETRY_COUNT):
            try:
                # 在重试前添加随机延迟，避免多个请求同时重试造成的冲击
                if retry > 0:
                    # 指数退避策略：INITIAL_RETRY_INTERVAL * (RETRY_INTERVAL_MULTIPLIER^retry) + 随机抖动
                    delay = INITIAL_RETRY_INTERVAL * (RETRY_INTERVAL_MULTIPLIER ** retry) + random.uniform(0, 1)
                    logger.info(f"第{retry}次重试，等待{delay:.2f}秒...")
                    print(f"第{retry}次重试，等待{delay:.2f}秒...")
                    time.sleep(delay)
                
                # 根据配置选择不同的LLM服务
                if LLM_SERVICE == "OPENAI":
                    logger.info("使用OpenAI服务进行分类...")
                    print("使用OpenAI服务进行分类...")
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    response = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": final_prompt}],
                        temperature=TEMPERATURE,
                        timeout=API_TIMEOUT,  # 使用配置的超时时间
                        stream=False
                    )
                elif LLM_SERVICE == "DEEPSEEK":
                    logger.info("使用DeepSeek服务进行分类...")
                    print("使用DeepSeek服务进行分类...")
                    client = OpenAI(
                        api_key=DEEPSEEK_API_KEY,
                        base_url=DEEPSEEK_BASE_URL
                    )
                    response = client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=[{"role": "user", "content": final_prompt}],
                        temperature=TEMPERATURE,
                        timeout=API_TIMEOUT,  # 使用配置的超时时间
                        stream=False
                    )
                elif LLM_SERVICE == "LOCAL":
                    logger.info("使用本地LLM服务进行分类...")
                    print("使用本地LLM服务进行分类...")
                    client = LocalLLMClient()
                    # 使用chat方法发送请求，直接返回OpenAI格式相似的响应对象
                    response = client.chat(user_content=final_prompt)
                else:
                    error_msg = f"不支持的模型服务: {LLM_SERVICE}"
                    logger.error(error_msg)
                    print(error_msg)
                    return None
                    
                # 验证响应格式
                if not hasattr(response, 'choices') or not response.choices or \
                   not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
                    raise ValueError("LLM返回的响应格式无效")
                
                # 解析LLM输出
                result_lines = response.choices[0].message.content.strip().split("\n")
                classify_data = []
                
                for line in result_lines:
                    # 按"\t"分割（严格匹配输出格式）
                    parts = line.split("\t")
                    if len(parts) == 1:
                        parts = line.split("\\t")
                    if len(parts) == 4:
                        classify_data.append({
                            "raw_text": parts[0].strip(),
                            "category": parts[1].strip(),
                            "confidence": parts[2].strip(),
                            "reason": parts[3].strip()
                        })
                
                logger.info(f"成功解析LLM响应，获得 {len(classify_data)} 个分类结果")
                
                # 合并原始数据与分类结果
                result_df = pd.merge(batch_df, pd.DataFrame(classify_data), on="raw_text", how="left")
                
                # 过滤掉未分类、无法识别和空分类的条目
                filtered_df = result_df[(
                    ~result_df["category"].str.contains("未分类|无法识别", na=False) & 
                    result_df["category"].notna() & 
                    result_df["category"].str.strip() != ""
                )].copy()
                
                logger.info(f"过滤掉未分类、无法识别和空分类条目，原记录数: {len(result_df)}, 过滤后记录数: {len(filtered_df)}")
                print(f"过滤掉未分类、无法识别和空分类条目，原记录数: {len(result_df)}, 过滤后记录数: {len(filtered_df)}")
                
                return filtered_df
                
            except Timeout as e:
                error_msg = f"LLM调用超时（尝试 {retry+1}/{MAX_RETRY_COUNT}）！错误：{str(e)}"
                logger.warning(error_msg)
                print(error_msg)
            except APIError as e:
                error_msg = f"LLM API错误（尝试 {retry+1}/{MAX_RETRY_COUNT}）！错误：{str(e)}"
                logger.warning(error_msg)
                print(error_msg)
            except Exception as e:
                error_msg = f"LLM调用失败（尝试 {retry+1}/{MAX_RETRY_COUNT}）！错误：{type(e).__name__} - {str(e)}"
                logger.warning(error_msg)
                print(error_msg)
            
            # 判断是否是最后一次重试
            if retry == MAX_RETRY_COUNT - 1:
                error_msg = f"达到最大重试次数，请求失败。"
                logger.error(error_msg)
                print(error_msg)
                return None
            # 继续重试
            continue
            
    except Exception as e:
        error_msg = f"处理批次文件 {batch_file_path} 时发生错误: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)
        return None

def batch_classify():
    """
    批量分类主函数
    
    功能：处理所有批次文件，使用多线程并行分类，合并结果
    处理流程：
    1. 创建并清空分类结果文件夹
    2. 加载Prompt模板
    3. 获取所有批次文件
    4. 使用线程池并行处理所有批次
    5. 合并所有分类结果
    """
    try:
        start_time = datetime.now()
        logger.info("开始批量分类处理")
        
        # 1. 创建分类结果文件夹
        logger.info(f"准备创建分类结果文件夹：{CLASSIFY_SAVE_PATH}")
        if not os.path.exists(CLASSIFY_SAVE_PATH):
            os.makedirs(CLASSIFY_SAVE_PATH)
            logger.info(f"已创建分类结果文件夹：{CLASSIFY_SAVE_PATH}")

        # 2. 删除CLASSIFY_SAVE_PATH下所有文件
        logger.info(f"清理分类结果文件夹中的旧文件")
        files_deleted = 0
        for filename in os.listdir(CLASSIFY_SAVE_PATH):
            file_path = os.path.join(CLASSIFY_SAVE_PATH, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    files_deleted += 1
            except Exception as e:
                logger.error(f"删除文件 {file_path} 失败！错误：{e}")
        logger.info(f"清理完成，共删除 {files_deleted} 个旧文件")

        # 3. 加载Prompt模板
        prompt_template = load_prompt_template()
        if not prompt_template:
            logger.error("无法加载提示词模板，终止处理")
            return

        # 4. 获取所有批次文件（仅CSV）
        batch_files = [f for f in os.listdir(BATCH_SAVE_PATH) if f.endswith(".csv")]
        if not batch_files:
            warning_msg = f"未找到批次文件！请先运行script/data_preprocess.py"
            logger.warning(warning_msg)
            print(warning_msg)
            return
        
        logger.info(f"共找到{len(batch_files)}个批次文件，开始分类...")
        print(f"共找到{len(batch_files)}个批次文件，开始分类...")

        # 5. 定义单个批次处理函数（用于多线程）
        def process_batch(batch_file):
            batch_path = os.path.join(BATCH_SAVE_PATH, batch_file)
            thread_name = os.path.basename(batch_file)
            logger.info(f"线程 {thread_name} 开始处理...")
            print(f"线程 {thread_name} 开始处理...")
            
            try:
                # 分类当前批次
                result_df = classify_single_batch(batch_path, prompt_template)
                if result_df is None:
                    logger.warning(f"跳过{batch_file}（处理失败）")
                    print(f"跳过{batch_file}（处理失败）")
                    return False
                
                # 检查数据框是否为空（只有表头没有实际数据行）
                if len(result_df) == 0:
                    logger.info(f"跳过{batch_file}（分类结果为空，不生成文件）")
                    print(f"跳过{batch_file}（分类结果为空，不生成文件）")
                    return True  # 返回True表示处理成功，只是没有生成文件
                
                # 保存分类结果
                result_filename = f"result_{batch_file}"
                result_filepath = os.path.join(CLASSIFY_SAVE_PATH, result_filename)
                
                try:
                    result_df.to_csv(result_filepath, index=False, encoding="utf-8")
                    logger.info(f"已保存{result_filename}，包含 {len(result_df)} 条记录")
                    print(f"已保存{result_filename}")
                    return True
                except Exception as e:
                    logger.error(f"保存{result_filename}失败！错误：{e}")
                    print(f"保存{result_filename}失败！错误：{e}")
                    return False
            except Exception as e:
                logger.error(f"处理{batch_file}时发生异常：{e}")
                logger.error(traceback.format_exc())
                print(f"处理{batch_file}时发生异常：{e}")
                return False
        
        # 6. 使用线程池并行处理所有批次
        # 根据配置获取当前服务的并发限制
        concurrency_limit = LLM_CONCURRENCY.get(LLM_SERVICE, 2)
        
        # 根据CPU核心数、服务限制和批次文件数量动态调整线程数
        max_workers = min(os.cpu_count(), concurrency_limit, len(batch_files))  # 保守设置，优先使用服务限制
        
        logger.info(f"使用多线程处理，最大线程数：{max_workers}（基于{LLM_SERVICE}服务的并发限制）")
        print(f"使用多线程处理，最大线程数：{max_workers}（基于{LLM_SERVICE}服务的并发限制）")
        
        success_count = 0
        failed_count = 0
        
        # 检查是否有可用的批次文件
        if not batch_files:
            logger.warning("没有需要处理的批次文件")
            print("没有需要处理的批次文件")
            return
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务到线程池
            future_to_batch = {executor.submit(process_batch, batch_file): batch_file for batch_file in batch_files}
            
            # 等待所有任务完成并获取结果
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_file = future_to_batch[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"获取{batch_file}结果时发生异常：{e}")
                    print(f"处理{batch_file}时发生异常：{e}")
                    failed_count += 1

        # 记录处理结果统计
        logger.info(f"多线程处理完成！成功：{success_count} 个批次，失败：{failed_count} 个批次")
        print(f"多线程处理完成！成功：{success_count} 个批次，失败：{failed_count} 个批次")
        print(f"结果保存在：{CLASSIFY_SAVE_PATH}")
        
        # 合并所有分类结果文件
        logger.info("开始合并所有分类结果文件")
        merge_classification_results()
        
        # 输出总处理时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"批量分类处理完成，总耗时：{duration:.2f} 秒")
        print(f"批量分类处理完成，总耗时：{duration:.2f} 秒")
        
    except Exception as e:
        error_msg = f"批量分类过程中发生未预期错误！错误：{type(e).__name__} - {str(e)}"
        logger.critical(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)

# 执行批量分类
if __name__ == "__main__":
    batch_classify()