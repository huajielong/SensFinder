import pandas as pd
import os
import concurrent.futures
import time
import random
import glob
from openai import OpenAI

# 使用正确的包导入方式
from config import (
    BATCH_SAVE_PATH, CLASSIFY_SAVE_PATH, 
    CURRENT_LLM_SERVICE, LLM_TEMPERATURE,
    OPENAI_API_KEY, OPENAI_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
)
from script.local_llm import LocalLLM

# 合并分类结果文件的函数
def merge_classification_results():
    """
    合并data/classify_results目录下的所有CSV文件到一个总表中，并按category字段排序
    结果保存到data/merged_results.csv
    """
    # 使用相对路径，避免硬编码
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(project_root, 'data', 'classify_results')
    output_file = os.path.join(project_root, 'data', 'merged_results.csv')
    
    print(f"\n开始合并分类结果文件...")
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"没有找到CSV文件在目录: {input_dir}")
        return
    
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
            
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    
    # 检查是否成功合并了数据
    if all_data.empty:
        print("警告: 没有成功合并任何数据!")
        return
    
    print(f"合并完成! 总数据行数: {len(all_data)}")
    
    # 按category字段排序
    if 'category' in all_data.columns:
        all_data = all_data.sort_values(by='category', ignore_index=True)
        print("已按 category 字段排序")
    else:
        print("警告: 数据中没有 'category' 字段，无法排序")
    
    # 保存合并后的数据
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        all_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"合并结果已保存到: {output_file}")
        
        # 显示一些基本统计信息
        print("\n数据统计信息:")
        print(f"总列数: {len(all_data.columns)}")
        print(f"总行数: {len(all_data)}")
        
        if 'category' in all_data.columns:
            print("\nCategory 分布:")
            category_counts = all_data['category'].value_counts()
            print(category_counts)
            
            print(f"\n唯一 category 数量: {len(category_counts)}")
            
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")

# 加载LLM提示词模板
def load_prompt_template():
    # 使用项目根目录相对路径
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "prompt_template.txt")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"读取Prompt模板失败！错误：{e}")
        print(f"请检查 {template_path} 是否存在")
        return None

# 调用LLM分类单个批次（带重试机制）
def classify_single_batch(batch_file_path, prompt_template):
    # 读取批次文件
    batch_df = pd.read_csv(batch_file_path, encoding="utf-8")
    fields = batch_df["raw_text"].tolist()
    
    # 格式化字段为"1. 字段1\n2. 字段2"格式
    fields_text = "\n".join([f"{i+1}. {field}" for i, field in enumerate(fields)])
    # 填充Prompt模板
    final_prompt = prompt_template.replace("{{fields_text}}", fields_text)
    
    # 重试配置
    max_retries = 5  # 最大重试次数
    base_delay = 1   # 基础延迟时间（秒）
    
    # 执行带重试的LLM调用
    for retry in range(max_retries):
        try:
            # 在重试前添加随机延迟，避免多个请求同时重试造成的冲击
            if retry > 0:
                # 指数退避策略：base_delay * (2^retry) + 随机抖动
                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                print(f"第{retry}次重试，等待{delay:.2f}秒...")
                time.sleep(delay)
            
            if CURRENT_LLM_SERVICE == "OPENAI":
                print("使用OpenAI服务进行分类...")
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=LLM_TEMPERATURE,
                    timeout=60,
                    stream=False
                )
            elif CURRENT_LLM_SERVICE == "DEEPSEEK":
                print("使用DeepSeek服务进行分类...")
                client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL
                )
                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=LLM_TEMPERATURE,
                    timeout=60,
                    stream=False
                )
            elif CURRENT_LLM_SERVICE == "LOCAL":
                print("使用本地LLM服务进行分类...")
                client = LocalLLM()
                # 使用chat方法发送请求，直接返回OpenAI格式相似的响应对象
                response = client.chat(user_content=final_prompt)
            else:
                print(f"不支持的模型服务: {CURRENT_LLM_SERVICE}")
                return None
                
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
            # 合并原始数据与分类结果
            result_df = pd.merge(batch_df, pd.DataFrame(classify_data), on="raw_text", how="left")
            
            # 过滤掉未分类、无法识别和空分类的条目
            filtered_df = result_df[(~result_df["category"].str.contains("未分类|无法识别", na=False)) & (result_df["category"].notna()) & (result_df["category"].str.strip() != "")].copy()
            print(f"过滤掉未分类、无法识别和空分类条目，原记录数: {len(result_df)}, 过滤后记录数: {len(filtered_df)}")
            
            return filtered_df
            
        except Exception as e:
            print(f"LLM调用失败（尝试 {retry+1}/{max_retries}）！错误：{type(e).__name__} - {str(e)}")
            # 判断是否是最后一次重试
            if retry == max_retries - 1:
                print(f"达到最大重试次数，请求失败。")
                return None
            # 继续重试
            continue

# 批量处理所有批次
def batch_classify():
    # 1. 创建分类结果文件夹
    if not os.path.exists(CLASSIFY_SAVE_PATH):
        os.makedirs(CLASSIFY_SAVE_PATH)
        print(f"已创建分类结果文件夹：{CLASSIFY_SAVE_PATH}")

    # 2. 删除CLASSIFY_SAVE_PATH下所有文件
    for filename in os.listdir(CLASSIFY_SAVE_PATH):
        file_path = os.path.join(CLASSIFY_SAVE_PATH, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"已删除文件：{file_path}")
        except Exception as e:
            print(f"删除文件 {file_path} 失败！错误：{e}")

    # 3. 加载Prompt模板
    prompt_template = load_prompt_template()
    if not prompt_template:
        return

    # 4. 获取所有批次文件（仅CSV）
    batch_files = [f for f in os.listdir(BATCH_SAVE_PATH) if f.endswith(".csv")]
    if not batch_files:
        print(f"未找到批次文件！请先运行script/data_preprocess.py")
        return
    print(f"共找到{len(batch_files)}个批次文件，开始分类...")

    # 5. 定义单个批次处理函数（用于多线程）
    def process_batch(batch_file):
        batch_path = os.path.join(BATCH_SAVE_PATH, batch_file)
        print(f"线程 {batch_file} 开始处理...")
        
        # 分类当前批次
        result_df = classify_single_batch(batch_path, prompt_template)
        if result_df is None:
            print(f"跳过{batch_file}（处理失败）")
            return False
        
        # 检查数据框是否为空（只有表头没有实际数据行）
        if len(result_df) == 0:
            print(f"跳过{batch_file}（分类结果为空，不生成文件）")
            return True  # 返回True表示处理成功，只是没有生成文件
        
        # 保存分类结果
        result_filename = f"result_{batch_file}"
        result_df.to_csv(os.path.join(CLASSIFY_SAVE_PATH, result_filename), index=False, encoding="utf-8")
        print(f"已保存{result_filename}")
        return True
    
    # 6. 使用线程池并行处理所有批次
    # 控制并发请求数量，避免服务过载
    # 根据不同LLM服务设置合理的最大并发数
    service_concurrency_limit = {
        "OPENAI": 2,     # OpenAI API的并发限制
        "DEEPSEEK": 3,   # DeepSeek API的并发限制
        "LOCAL": 4       # 本地LLM的并发限制
    }
    
    # 获取当前服务的并发限制
    concurrency_limit = service_concurrency_limit.get(CURRENT_LLM_SERVICE, 2)
    # 根据CPU核心数、服务限制和批次文件数量动态调整线程数
    max_workers = min(os.cpu_count(), concurrency_limit, len(batch_files))  # 保守设置，优先使用服务限制
    print(f"使用多线程处理，最大线程数：{max_workers}（基于{CURRENT_LLM_SERVICE}服务的并发限制）")
    
    success_count = 0
    failed_count = 0
    
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
                print(f"处理{batch_file}时发生异常：{e}")
                failed_count += 1

    
    print(f"多线程处理完成！成功：{success_count} 个批次，失败：{failed_count} 个批次")
    print(f"结果保存在：{CLASSIFY_SAVE_PATH}")
    
    # 合并所有分类结果文件
    merge_classification_results()



# 执行批量分类
if __name__ == "__main__":
    batch_classify()