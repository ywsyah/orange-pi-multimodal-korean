import json
import os
from PIL import Image
import shutil

import argparse

# 配置信息
input_json_path = r'E:\Desktop\ai相关资料\学习资料路线等\中科院软件实习项目\项目工程文件测试\数据集\OpenDataLab___WanJuanSiLu2O\raw\image\ko\converted_mllm_data_local_ko_all.json'  # 原始JSON文件路径
output_json_path = r'E:\Desktop\ai相关资料\学习资料路线等\中科院软件实习项目\项目工程文件测试\数据集\OpenDataLab___WanJuanSiLu2O\raw\image\ko\converted_mllm_data_local_ko_clean_all.json'  # 新的JSON文件路径（不含损坏图片的记录）
backup_dir = 'damaged_files_backup'  # 损坏文件备份目录（删除前备份，以防万一）

def main(input_json_path, output_json_path, backup_dir):
    # 创建备份目录（如果不存在）
    os.makedirs(backup_dir, exist_ok=True)

    # 存储损坏的文件路径
    damaged_files = []

    # 1. 第一遍扫描：识别损坏的文件
    print("第一阶段: 扫描识别损坏的图片文件...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                
                # 提取图片路径
                if 'image' in data and 'path' in data['image']:
                    image_path = data['image']['path']
                    
                    # 检查文件是否存在
                    if not os.path.exists(image_path):
                        print(f"文件不存在: {image_path}")
                        damaged_files.append(image_path)
                        continue
                    
                    # 尝试打开图片检查是否损坏
                    try:
                        with Image.open(image_path) as img:
                            # 验证图片完整性
                            img.verify()
                    except Exception as e:
                        print(f"发现损坏的文件: {image_path}，错误: {str(e)}")
                        damaged_files.append(image_path)
                        
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {str(e)}")

    # 打印损坏文件汇总信息
    print(f"\n共发现{len(damaged_files)}个损坏的文件")
    for path in damaged_files:
        print(f" - {path}")

    # 2. 备份并删除损坏的文件
    print("\n第二阶段: 备份并删除损坏的文件...")
    for damaged_path in damaged_files:
        try:
            # 确保备份目录的父目录结构存在
            backup_path = os.path.join(backup_dir, damaged_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # 如果文件存在，尝试备份
            if os.path.exists(damaged_path):
                # 复制到备份目录
                shutil.copy2(damaged_path, backup_path)
                print(f"已备份: {damaged_path} -> {backup_path}")
                
                # 删除原文件
                os.remove(damaged_path)
                print(f"已删除: {damaged_path}")
            else:
                print(f"文件不存在，无需备份和删除: {damaged_path}")
        except Exception as e:
            print(f"处理文件 {damaged_path} 时出错: {str(e)}")

    # 3. 创建新的JSON文件，排除引用损坏图片的记录
    print("\n第三阶段: 创建清理后的JSON文件...")
    damaged_set = set(damaged_files)  # 转换为集合以加快查找速度
    clean_count = 0
    damaged_count = 0

    with open(input_json_path, 'r', encoding='utf-8') as fin, open(output_json_path, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                
                # 检查是否引用了损坏的图片
                if 'image' in data and 'path' in data['image']:
                    image_path = data['image']['path']
                    
                    if image_path in damaged_set:
                        # 跳过引用损坏图片的记录
                        damaged_count += 1
                        continue
                
                # 写入未损坏的记录到新文件
                fout.write(line)
                clean_count += 1
                    
            except json.JSONDecodeError:
                # 如果JSON解析失败，跳过该行
                damaged_count += 1
                continue

    # 4. 最终报告
    print("\n=== 清理完成 ===")
    print(f"原始文件: {input_json_path}")
    print(f"清理后文件: {output_json_path}")
    print(f"保留记录数: {clean_count}")
    print(f"移除记录数: {damaged_count}")
    print(f"删除的损坏图片: {len(damaged_files)}")
    print(f"损坏图片备份目录: {backup_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='清理JSON文件中包含损坏图片的记录')
    parser.add_argument('--input_json_path', type=str, required=True, help='原始JSON文件路径')
    parser.add_argument('--output_json_path', type=str, required=True, help='清理后的JSON文件保存路径')
    parser.add_argument('--backup_dir', type=str, default='damaged_files_backup', help='损坏图片备份目录')

    args = parser.parse_args()

    main(args.input_json_path, args.output_json_path, args.backup_dir)
