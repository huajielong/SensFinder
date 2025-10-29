import pandas as pd
import os

# 使用包导入方式
from config import RAW_FILES_PATH, BATCH_SAVE_PATH, BATCH_SIZE, MIN_FIELD_LENGTH

def preprocess_data():
    # 1. 创建输出文件夹（不存在则创建）
    if not os.path.exists(BATCH_SAVE_PATH):
        os.makedirs(BATCH_SAVE_PATH)
        print(f"已创建批次文件保存文件夹：{BATCH_SAVE_PATH}")

    # 2. 读取原始文本文件（递归处理目录下所有文件）
    try:
        raw_fields = []
        total_files = 0
        
        # 检查路径是否存在
        if not os.path.exists(RAW_FILES_PATH):
            print(f"错误：目录 {RAW_FILES_PATH} 不存在")
            return
            
        # 递归遍历目录下所有文件
        for root, dirs, files in os.walk(RAW_FILES_PATH):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        # 读取当前文件的所有非空行，并按空格分割获取多个字段
                        file_fields = []
                        for line in f:
                            stripped_line = line.strip()
                            if stripped_line:
                                # 按空格分割，获取多个字段
                                fields = stripped_line.split()
                                file_fields.extend(fields)
                        raw_fields.extend(file_fields)
                    total_files += 1
                    print(f"已读取文件：{file_path}，找到 {len(file_fields)} 个字段")
                except Exception as e:
                    print(f"读取文件 {file_path} 失败！错误：{e}，跳过该文件")
        
        if total_files == 0:
            print(f"警告：目录 {RAW_FILES_PATH} 下没有找到可读取的文件")
            return
            
        print(f"成功读取 {total_files} 个文件，共收集到 {len(raw_fields)} 个字段")
    except Exception as e:
        print(f"处理目录时发生错误！错误：{e}")
        print(f"请检查 config.py 中 RAW_FILES_PATH 是否正确")
        return

    # 3. 清理无效字段（长度＜MIN_FIELD_LENGTH、纯特殊字符）
    import re
    # 过滤规则：长度≥MIN_FIELD_LENGTH + 至少含1个字母（排除纯数字/特殊字符）
    valid_fields = []
    for field in raw_fields:
        if len(field) >= MIN_FIELD_LENGTH and re.search(r'[a-zA-Z]', field):
            valid_fields.append(field)
    print(f"清理后有效字段数：{len(valid_fields)}（删除{len(raw_fields)-len(valid_fields)}个无效字段）")

    # 4. 去重（确保无重复字段）
    valid_fields = list(set(valid_fields))
    print(f"去重后最终字段数：{len(valid_fields)}")

    # 5. 分批次保存为CSV
    total_batches = len(valid_fields) // BATCH_SIZE + 1
    for batch_idx in range(total_batches):
        # 计算当前批次的字段范围
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_fields = valid_fields[start:end]
        
        # 保存为CSV（含raw_text列）
        batch_df = pd.DataFrame({"raw_text": batch_fields})
        batch_filename = f"batch_{batch_idx+1}.csv"
        batch_df.to_csv(os.path.join(BATCH_SAVE_PATH, batch_filename), index=False, encoding="utf-8")
    
    print(f"预处理完成！共生成{total_batches}个批次文件，保存在：{BATCH_SAVE_PATH}")

# 执行预处理
if __name__ == "__main__":
    preprocess_data()