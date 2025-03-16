# 预处理离线工程配置
import json5  # 解决复杂json场景
import sys
import os
import json
import argparse

# 解析 uniapp  manifest.json 文件
def parse_uniapp_manifest(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json5.load(f)
            version_name = data.get('versionName')
            version_code = data.get('versionCode')
            return version_name, version_code
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except json5.JSONDecodeError:
        print("错误：文件内容不符合JSON格式")
    except KeyError as e:
        print(f"错误：未找到字段 {e}")

def generate_manifest(version_code, version_name, output_file="manifest.json"):
    # 构建 JSON 数据结构
    manifest_data = {
        "versionCode": version_code,
        "versionName": version_name
    }
    # 写入文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=4, ensure_ascii=False)
        print(f"成功生成 {output_file}")
    except Exception as e:
        print(f"文件写入失败: {str(e)}")

# def modify_via_ast(ruby_file_path, jsonString):
#   subprocess.run([
#     'ruby', 'pack/offline/ast.rb',
#     ruby_file_path, jsonString
#   ])

if __name__ == '__main__':
    remove_path = sys.argv[1]
    print("提取manifest.json信息")
    file_path = "./src/manifest.json"
    version_name, version_code = parse_uniapp_manifest(file_path)
    print(f"版本名称: {version_name}, "
          f"版本号: {version_code}")
    generate_manifest(version_code,version_name, remove_path+'/manifest.json')
    print("manifest.json新路径:", remove_path+'/manifest.json')