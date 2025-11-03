import pandas as pd
import os
import re
import logging
import traceback
from datetime import datetime
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从config.config正确导入配置
from config.config import (
    CLASSIFY_SAVE_PATH,
    PROBLEM_SAVE_PATH,
    LOW_CONFIDENCE_THRESHOLD,
    COMPANY_NAME_KEYWORDS,
    PROJECT_ROOT
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs/result_verify.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_results():
    """
    验证分类结果主函数
    
    功能：验证所有分类结果，检测以下问题：
    1. 规则冲突：含公司关键词但分类不是公司名的字段
    2. 低置信度：置信度低于阈值的字段
    3. 空值/无效值：缺少类别或置信度的字段
    4. 异常格式：格式异常的字段
    
    结果保存到问题字段目录
    """
    try:
        start_time = datetime.now()
        logger.info("开始验证分类结果")
        
        # 1. 创建问题字段保存文件夹
        logger.info(f"准备创建问题字段文件夹：{PROBLEM_SAVE_PATH}")
        if not os.path.exists(PROBLEM_SAVE_PATH):
            try:
                os.makedirs(PROBLEM_SAVE_PATH)
                logger.info(f"已创建问题字段文件夹：{PROBLEM_SAVE_PATH}")
                print(f"已创建问题字段文件夹：{PROBLEM_SAVE_PATH}")
            except Exception as e:
                error_msg = f"创建问题字段文件夹失败：{str(e)}"
                logger.error(error_msg)
                print(error_msg)
                return

        # 2. 删除PROBLEM_SAVE_PATH下所有文件
        logger.info(f"清理问题字段文件夹中的旧文件")
        files_deleted = 0
        for filename in os.listdir(PROBLEM_SAVE_PATH):
            file_path = os.path.join(PROBLEM_SAVE_PATH, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    files_deleted += 1
            except Exception as e:
                logger.error(f"删除文件 {file_path} 失败：{str(e)}")
        logger.info(f"清理完成，共删除 {files_deleted} 个旧文件")

        # 3. 获取所有分类结果文件
        result_files = [f for f in os.listdir(CLASSIFY_SAVE_PATH) 
                       if f.startswith("result_") and f.endswith(".csv")]
        if not result_files:
            warning_msg = f"未找到分类结果文件！请先运行script/llm_classify.py"
            logger.warning(warning_msg)
            print(warning_msg)
            return
        
        logger.info(f"共找到{len(result_files)}个分类结果文件，开始验证...")
        print(f"共找到{len(result_files)}个分类结果文件，开始验证...")

        # 4. 逐个文件验证
        all_problems = []
        total_processed = 0
        
        for result_file in result_files:
            result_path = os.path.join(CLASSIFY_SAVE_PATH, result_file)
            logger.info(f"正在验证：{result_file}")
            print(f"正在验证：{result_file}")
            
            try:
                # 读取分类结果
                result_df = pd.read_csv(result_path, encoding="utf-8")
                logger.info(f"{result_file} 包含 {len(result_df)} 条记录")
                total_processed += len(result_df)
                
                # 填充空值（避免处理报错）
                result_df = result_df.fillna({
                    "category": "未分类", 
                    "confidence": 0, 
                    "reason": "无"
                })

                # -------------------------- 验证1：规则冲突（公司名关键词vs分类结果） --------------------------
                # 检查含公司关键词但分类不是"公司名"的字段
                if COMPANY_NAME_KEYWORDS and len(COMPANY_NAME_KEYWORDS) > 0:
                    company_keyword_pattern = "|".join([re.escape(keyword) for keyword in COMPANY_NAME_KEYWORDS])
                    try:
                        result_df["has_company_keyword"] = result_df["raw_text"].str.contains(
                            company_keyword_pattern, case=False, na=False
                        )
                        conflict_company = result_df[
                            (result_df["has_company_keyword"]) & 
                            (~result_df["category"].str.contains("公司名", na=False))
                        ].copy()
                        conflict_company["problem_type"] = "公司关键词冲突"
                        logger.info(f"{result_file} 发现 {len(conflict_company)} 个公司关键词冲突字段")
                    except Exception as e:
                        logger.error(f"{result_file} 公司关键词冲突检测失败：{str(e)}")
                        conflict_company = pd.DataFrame()
                else:
                    conflict_company = pd.DataFrame()

                # -------------------------- 验证2：低置信度（＜LOW_CONFIDENCE_THRESHOLD） --------------------------
                # 转换置信度为整数（处理可能的字符串格式）
                try:
                    result_df["confidence_int"] = pd.to_numeric(result_df["confidence"], errors="coerce").fillna(0)
                    low_conf = result_df[result_df["confidence_int"] < LOW_CONFIDENCE_THRESHOLD].copy()
                    low_conf["problem_type"] = "低置信度"
                    logger.info(f"{result_file} 发现 {len(low_conf)} 个低置信度字段")
                except Exception as e:
                    logger.error(f"{result_file} 低置信度检测失败：{str(e)}")
                    low_conf = pd.DataFrame()

                # -------------------------- 验证3：空值/无效值检测 --------------------------
                try:
                    # 检测空分类或无效分类
                    invalid_category = result_df[
                        (result_df["category"].str.strip() == "") | 
                        (result_df["category"].str.contains("未分类|无法识别", na=False))
                    ].copy()
                    invalid_category["problem_type"] = "无效分类"
                    logger.info(f"{result_file} 发现 {len(invalid_category)} 个无效分类字段")
                except Exception as e:
                    logger.error(f"{result_file} 无效分类检测失败：{str(e)}")
                    invalid_category = pd.DataFrame()
                
                # -------------------------- 验证4：异常格式检测 --------------------------
                try:
                    # 检测格式异常的字段
                    format_error = result_df[
                        (result_df["raw_text"].str.len() > 500) |  # 过长字段
                        (result_df["raw_text"].str.count("\\n") > 5)  # 多行字段
                    ].copy()
                    format_error["problem_type"] = "格式异常"
                    logger.info(f"{result_file} 发现 {len(format_error)} 个格式异常字段")
                except Exception as e:
                    logger.error(f"{result_file} 格式异常检测失败：{str(e)}")
                    format_error = pd.DataFrame()

                # -------------------------- 收集问题字段 --------------------------
                batch_problems = pd.concat(
                    [conflict_company, low_conf, invalid_category, format_error], 
                    ignore_index=True
                ).drop_duplicates(subset=["raw_text"])
                
                if not batch_problems.empty:
                    # 添加批次信息
                    batch_problems["source_batch"] = result_file
                    all_problems.append(batch_problems)
                    logger.info(f"{result_file} 总计发现 {len(batch_problems)} 个问题字段")
                    print(f"  该批次发现{len(batch_problems)}个问题字段")
                else:
                    logger.info(f"{result_file} 未发现问题字段")
                    print(f"  该批次未发现问题字段")
            except Exception as e:
                error_msg = f"处理文件 {result_file} 时发生错误：{str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                print(error_msg)

        # 5. 保存所有问题字段
        if all_problems:
            all_problems_df = pd.concat(all_problems, ignore_index=True)
            # 只保留关键列（便于人工复核）
            output_columns = ["source_batch", "raw_text", "category", "confidence", "reason", "problem_type"]
            output_path = os.path.join(PROBLEM_SAVE_PATH, "all_problems.csv")
            
            try:
                all_problems_df[output_columns].to_csv(
                    output_path, 
                    index=False, encoding="utf-8-sig"
                )
                logger.info(f"所有问题字段已保存到：{output_path}")
                print(f"\n所有问题字段已保存：{output_path}")
                print(f"总计问题字段数：{len(all_problems_df)}")
                
                # 按问题类型统计
                problem_stats = all_problems_df["problem_type"].value_counts()
                logger.info(f"问题类型统计：\n{problem_stats}")
                print(f"\n问题类型统计：")
                print(problem_stats)
                
                # 计算问题率
                if total_processed > 0:
                    problem_rate = (len(all_problems_df) / total_processed) * 100
                    logger.info(f"总处理记录数：{total_processed}，问题率：{problem_rate:.2f}%")
                    print(f"\n总处理记录数：{total_processed}")
                    print(f"问题率：{problem_rate:.2f}%")
                
                # 保存问题字段分类统计
                if not all_problems_df.empty:
                    category_stats = all_problems_df.groupby(["problem_type", "category"]).size().unstack(fill_value=0)
                    category_stats.to_csv(
                        os.path.join(PROBLEM_SAVE_PATH, "problem_category_stats.csv"),
                        encoding="utf-8-sig"
                    )
                    logger.info("问题字段分类统计已保存")
                    
            except Exception as e:
                error_msg = f"保存问题字段文件失败：{str(e)}"
                logger.error(error_msg)
                print(error_msg)
        else:
            logger.info("未发现问题字段！分类结果质量良好")
            print("\n未发现问题字段！分类结果质量良好")
        
        # 输出总处理时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"结果验证完成，总耗时：{duration:.2f} 秒")
        print(f"\n结果验证完成，总耗时：{duration:.2f} 秒")
        
    except Exception as e:
        error_msg = f"结果验证过程中发生未预期错误：{type(e).__name__} - {str(e)}"
        logger.critical(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)

# 执行验证
if __name__ == "__main__":
    verify_results()