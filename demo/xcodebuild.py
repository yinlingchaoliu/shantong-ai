import subprocess
import os
import plistlib
import sys
import json

# 项目要必须切到 catech-app-ios-mall 目录下 才可以执行

# 全局变量
SCHEME = "HBuilder"  # 替换为你的 Scheme
WORKSPACE = "CatechAppIosMall.xcworkspace"  # 替换为你的 Workspace 文件
CONFIGURATION = "Debug"  # 配置类型：Release 或 Debug
EXPORT_METHOD = "debugging"  # 导出方式：app-store, ad-hoc, enterprise, development ,debugging
EXPORT_PATH = "./build"  # 导出路径
EXPORT_PATH = os.path.expanduser(EXPORT_PATH)  # 构建路径
PROVISIONING_PROFILE = "cncomcartech_dev"  # 描述文件名称 cncomcartech_appstore
SIGNING_IDENTITY = "Apple Development: gaoxiong li (5AGS42ARSC)"  # 签名证书

# 创建导出路径
os.makedirs(os.path.expanduser(EXPORT_PATH), exist_ok=True)


# 切换证书
def switchCert(env):
    global CONFIGURATION  # 配置类型：Release 或 Debug
    global EXPORT_METHOD  # 导出方式：app-store, ad-hoc, enterprise, development ,debugging
    global PROVISIONING_PROFILE  # 描述文件名称 cncomcartech_appstore
    global SIGNING_IDENTITY  # 签名证书
    if env == "release":
        print("使用生产证书")
        CONFIGURATION = "Release"
        EXPORT_METHOD = "app-store"  # 导出方式：app-store, ad-hoc, enterprise, development ,debugging
        PROVISIONING_PROFILE = "cncomcartech_appstore"  # 描述文件名称 cncomcartech_appstore
        SIGNING_IDENTITY = "iPhone Distribution: Haier Electric Appliance International Co., Ltd. (KRV5T3384C)"  # 签名证书
    else:
        print("使用开发证书")
        CONFIGURATION = "Debug"
        EXPORT_METHOD = "debugging"  # 导出方式：app-store, ad-hoc, enterprise, development ,debugging
        PROVISIONING_PROFILE = "cncomcartech_dev"  # 描述文件名称 cncomcartech_appstore
        SIGNING_IDENTITY = "Apple Development: gaoxiong li (5AGS42ARSC)"  # 签名证书


# 读取json配置
def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 直接解析为 Python 字典
            return data
    except FileNotFoundError:
        print(f"错误：文件 {{file_path}} 不存在")
    except json.JSONDecodeError:
        print("错误：JSON 格式不合法")


# 更新plist版本号
def update_plist(json):
    # 读取json配置
    versionName = json['versionName']
    versionCode = json['versionCode']
    print("正在更新版本号...")
    # 文件路径
    plist_path = "CatechAppIosMall/CatechAppIosMall-Info.plist"
    # 读取 plist 文件
    with open(plist_path, "rb") as f:
        plist_data = plistlib.load(f)
    # 修改版本号和构建号
    plist_data["CFBundleShortVersionString"] = versionName  # 版本号
    plist_data["CFBundleVersion"] = versionCode  # 构建号
    # 保存修改后的 plist 文件
    with open(plist_path, "wb") as f:
        plistlib.dump(plist_data, f)
    print("版本号已更新！")


def archive():
    # 打包命令
    archive_command = [
        "xcodebuild",
        "archive",
        "-workspace", WORKSPACE,
        "-scheme", SCHEME,
        "-configuration", CONFIGURATION,
        "-archivePath", os.path.expanduser(f"{EXPORT_PATH}/{SCHEME}.xcarchive"),
    ]
    # 执行打包
    print("打包开始...")
    subprocess.run(archive_command, check=True)
    print("打包结束...")


def export():
    print("导出开始...")
    # 导出命令
    export_command = [
        "xcodebuild",
        "-exportArchive",
        "-archivePath", os.path.expanduser(f"{EXPORT_PATH}/{SCHEME}.xcarchive"),
        "-exportPath", os.path.expanduser(EXPORT_PATH),
        "-exportOptionsPlist", os.path.expanduser(f"{EXPORT_PATH}/exportOptions.plist"),
    ]
    print("导出exportOptions.plist")
    # 创建 exportOptions.plist  文件
    export_options = {
        "method": EXPORT_METHOD,
        "provisioningProfiles": {
            "cn.com.cartech": PROVISIONING_PROFILE,  # 替换为你的 Bundle ID
        },
        "signingStyle": "manual",
        "signingCertificate": SIGNING_IDENTITY,
    }
    with open(os.path.expanduser(f"{EXPORT_PATH}/exportOptions.plist"), "wb") as f:
        import plistlib
        plistlib.dump(export_options, f)
    print("导出执行...")
    # 执行导出
    subprocess.run(export_command, check=True)
    print("导出结束...")

# 项目要必须切到 catech-app-ios-mall 目录下 才可以执行
# debug | release
if __name__ == '__main__':
    env = sys.argv[1]
    json = read_json('manifest.json')
    # 采用json更新plist文件
    update_plist(json)
    # 切换证书
    switchCert(env)
    archive()
    export()
    print(f"打包和导出完成！文件已保存到：{EXPORT_PATH}")
