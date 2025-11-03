import pandas as pd
import os
import re
import logging
import traceback
from datetime import datetime

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从config.config正确导入配置
from config.config import (
    RAW_FILES_PATH,
    BATCH_SAVE_PATH,
    BATCH_SIZE,
    MIN_FIELD_LENGTH,
    PROJECT_ROOT
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs/data_preprocess.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_data():
    """
    数据预处理主函数
    
    功能：读取原始文本文件，提取、清洗、去重和划分字段批次，为后续LLM分类做准备
    
    处理流程：
    1. 创建并清空输出文件夹
    2. 检查原始数据目录是否存在
    3. 递归读取所有文本文件
    4. 处理文件编码问题
    5. 清洗和过滤无效字段
    6. 去重处理
    7. 分批次保存为CSV文件
    """
    try:
        start_time = datetime.now()
        logger.info(f"开始数据预处理，原始文件路径：{RAW_FILES_PATH}")
        
        # 1. 创建输出文件夹（不存在则创建）
        logger.info(f"准备创建输出文件夹：{BATCH_SAVE_PATH}")
        if not os.path.exists(BATCH_SAVE_PATH):
            os.makedirs(BATCH_SAVE_PATH)
            logger.info(f"已创建批次文件保存文件夹：{BATCH_SAVE_PATH}")
        
        # 2. 删除BATCH_SAVE_PATH下所有文件（清空旧文件）
        logger.info(f"清理输出文件夹中的旧文件")
        files_deleted = 0
        for filename in os.listdir(BATCH_SAVE_PATH):
            file_path = os.path.join(BATCH_SAVE_PATH, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    files_deleted += 1
            except Exception as e:
                logger.error(f"删除文件 {file_path} 失败！错误：{e}")
        logger.info(f"清理完成，共删除 {files_deleted} 个旧文件")

        # 3. 读取原始文本文件（递归处理目录下所有文件）
        raw_fields = []
        total_files = 0
        processed_files = 0
        failed_files = 0
        
        # 检查路径是否存在
        if not os.path.exists(RAW_FILES_PATH):
            error_msg = f"错误：目录 {RAW_FILES_PATH} 不存在"
            logger.error(error_msg)
            print(error_msg)
            return
            
        logger.info(f"开始扫描原始文件目录：{RAW_FILES_PATH}")
        # 递归遍历目录下所有文件
        for root, dirs, files in os.walk(RAW_FILES_PATH):
            for filename in files:
                total_files += 1
                file_path = os.path.join(root, filename)
                try:
                    # 尝试多种编码处理
                    file_fields = []
                    encodings = ['utf-8', 'latin-1', 'cp1252']
                    content_read = False
                    
                    for encoding in encodings:
                        try:
                            with open(file_path, "r", encoding=encoding) as f:
                                # 读取当前文件的所有非空行，并按空格分割获取多个字段
                                for line in f:
                                    stripped_line = line.strip()
                                    if stripped_line:
                                        # 按空格分割，获取多个字段
                                        fields = stripped_line.split()
                                        file_fields.extend(fields)
                            content_read = True
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if not content_read:
                        logger.warning(f"无法解码文件：{file_path}，跳过该文件")
                        failed_files += 1
                        continue
                    
                    raw_fields.extend(file_fields)
                    processed_files += 1
                    logger.info(f"已读取文件：{file_path}，找到 {len(file_fields)} 个字段")
                except Exception as e:
                    logger.error(f"读取文件 {file_path} 失败！错误：{e}，跳过该文件")
                    failed_files += 1
        
        if total_files == 0:
            warning_msg = f"警告：目录 {RAW_FILES_PATH} 下没有找到可读取的文件"
            logger.warning(warning_msg)
            print(warning_msg)
            return
            
        logger.info(f"文件扫描完成 - 总计: {total_files} 个文件, 成功: {processed_files} 个, 失败: {failed_files} 个")
        logger.info(f"共收集到 {len(raw_fields)} 个原始字段")

        # 4. 清理无效字段（长度＜MIN_FIELD_LENGTH、纯特殊字符）
        logger.info(f"开始清理无效字段，最小长度要求：{MIN_FIELD_LENGTH}")
        # 过滤规则：长度≥MIN_FIELD_LENGTH + 至少含1个字母或数字（排除纯特殊字符）
        valid_fields = []
        for field in raw_fields:
            if (len(field) >= MIN_FIELD_LENGTH and 
                re.search(r'[a-zA-Z0-9]', field)):
                valid_fields.append(field)
        
        removed_fields = len(raw_fields) - len(valid_fields)
        logger.info(f"清理完成 - 有效字段数：{len(valid_fields)}，删除了 {removed_fields} 个无效字段")

        # 5. 去重（确保无重复字段）
        logger.info("开始去重处理")
        unique_fields = list(set(valid_fields))
        duplicate_count = len(valid_fields) - len(unique_fields)
        logger.info(f"去重完成 - 最终字段数：{len(unique_fields)}，移除了 {duplicate_count} 个重复字段")

        # 6. 分批次保存为CSV
        logger.info(f"开始分批次保存，每批次大小：{BATCH_SIZE}")
        total_batches = (len(unique_fields) + BATCH_SIZE - 1) // BATCH_SIZE
        batches_created = 0
        
        for batch_idx in range(total_batches):
            try:
                # 计算当前批次的字段范围
                start = batch_idx * BATCH_SIZE
                end = start + BATCH_SIZE
                batch_fields = unique_fields[start:end]
                
                # 保存为CSV（含raw_text列）
                batch_df = pd.DataFrame({"raw_text": batch_fields})
                batch_filename = f"batch_{batch_idx+1}.csv"
                batch_filepath = os.path.join(BATCH_SAVE_PATH, batch_filename)
                
                batch_df.to_csv(batch_filepath, index=False, encoding="utf-8")
                batches_created += 1
                logger.info(f"已保存批次 {batch_idx+1}/{total_batches}：{len(batch_fields)} 个字段")
            except Exception as e:
                logger.error(f"保存批次 {batch_idx+1} 失败！错误：{e}")
        
        # 7. 输出统计信息
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"预处理完成！")
        logger.info(f"统计信息：")
        logger.info(f"- 原始字段总数：{len(raw_fields)}")
        logger.info(f"- 清理后字段数：{len(valid_fields)}")
        logger.info(f"- 去重后字段数：{len(unique_fields)}")
        logger.info(f"- 生成批次文件数：{batches_created}/{total_batches}")
        logger.info(f"- 总处理时间：{duration:.2f} 秒")
        logger.info(f"- 结果保存路径：{BATCH_SAVE_PATH}")
        
        print(f"预处理完成！共生成{batches_created}个批次文件，保存在：{BATCH_SAVE_PATH}")
        print(f"总处理时间：{duration:.2f} 秒")
        
    except Exception as e:
        error_msg = f"预处理过程中发生未预期错误！错误：{type(e).__name__} - {str(e)}"
        logger.critical(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)

# 执行预处理
if __name__ == "__main__":
    preprocess_data()