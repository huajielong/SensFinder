import pandas as pd
import os
import re

# 使用包导入方式
from config import CLASSIFY_SAVE_PATH, PROBLEM_SAVE_PATH, LOW_CONFIDENCE_THRESHOLD, COMPANY_KEYWORDS

def verify_results():
    # 1. 创建问题字段保存文件夹
    if not os.path.exists(PROBLEM_SAVE_PATH):
        os.makedirs(PROBLEM_SAVE_PATH)
        print(f"已创建问题字段文件夹：{PROBLEM_SAVE_PATH}")

    # 2. 删除PROBLEM_SAVE_PATH下所有文件
    for filename in os.listdir(PROBLEM_SAVE_PATH):
        file_path = os.path.join(PROBLEM_SAVE_PATH, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"已删除文件：{file_path}")
        except Exception as e:
            print(f"删除文件 {file_path} 失败！错误：{e}")

    # 3. 获取所有分类结果文件
    result_files = [f for f in os.listdir(CLASSIFY_SAVE_PATH) if f.startswith("result_") and f.endswith(".csv")]
    if not result_files:
        print(f"未找到分类结果文件！请先运行 02_llm_classify.py")
        return
    print(f"共找到{len(result_files)}个分类结果文件，开始验证...")

    # 4. 逐个文件验证
    all_problems = []
    for result_file in result_files:
        result_path = os.path.join(CLASSIFY_SAVE_PATH, result_file)
        print(f"正在验证：{result_file}")
        
        # 读取分类结果
        result_df = pd.read_csv(result_path, encoding="utf-8")
        # 填充空值（避免处理报错）
        result_df = result_df.fillna({"category": "未分类", "confidence": 0, "reason": "无"})

        # -------------------------- 验证1：规则冲突（公司名关键词vs分类结果） --------------------------
        # 检查含公司关键词但分类不是“公司名”的字段
        company_keyword_pattern = "|".join(COMPANY_KEYWORDS)
        result_df["has_company_keyword"] = result_df["raw_text"].str.contains(
            company_keyword_pattern, case=False, na=False
        )
        conflict_company = result_df[
            (result_df["has_company_keyword"]) & 
            (~result_df["category"].str.contains("公司名", na=False))
        ].copy()
        conflict_company["problem_type"] = "公司关键词冲突"

        # -------------------------- 验证2：低置信度（＜LOW_CONFIDENCE_THRESHOLD） --------------------------
        # 转换置信度为整数（处理可能的字符串格式）
        result_df["confidence_int"] = pd.to_numeric(result_df["confidence"], errors="coerce").fillna(0)
        low_conf = result_df[result_df["confidence_int"] < LOW_CONFIDENCE_THRESHOLD].copy()
        low_conf["problem_type"] = "低置信度"

        # -------------------------- 收集问题字段 --------------------------
        batch_problems = pd.concat([conflict_company, low_conf]).drop_duplicates(subset=["raw_text"])
        if not batch_problems.empty:
            # 添加批次信息
            batch_problems["source_batch"] = result_file
            all_problems.append(batch_problems)
            print(f"  该批次发现{len(batch_problems)}个问题字段")

    # 5. 保存所有问题字段
    if all_problems:
        all_problems_df = pd.concat(all_problems, ignore_index=True)
        # 只保留关键列（便于人工复核）
        output_columns = ["source_batch", "raw_text", "category", "confidence", "reason", "problem_type"]
        all_problems_df[output_columns].to_csv(
            os.path.join(PROBLEM_SAVE_PATH, "all_problems.csv"), 
            index=False, encoding="utf-8"
        )
        print(f"\n所有问题字段已保存：{PROBLEM_SAVE_PATH}/all_problems.csv")
        print(f"总计问题字段数：{len(all_problems_df)}")
    else:
        print("\n未发现问题字段！分类结果质量良好")

# 执行验证
if __name__ == "__main__":
    verify_results()