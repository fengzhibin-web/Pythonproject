import subprocess
import os

def run_script(script_name):
    """运行指定的 Python 脚本"""
    print(f"正在运行脚本: {script_name}")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"脚本 {script_name} 运行失败，错误信息如下：")
        print(result.stderr)
        raise RuntimeError(f"脚本 {script_name} 运行失败")
    print(f"脚本 {script_name} 运行成功")

def main():
    try:
        # 1. 数据清洗
        run_script("D:/Pythonproject/task04/final/optimization_ver\data_clean.py")

        # 2. 生成 GRA
        run_script("D:/Pythonproject/task04/final/optimization_ver/Image recognition system_generateGRA_ver02.py")

        # 3. 模型训练
        run_script("D:/Pythonproject/task04/final/optimization_ver/Image recognition system_train_ver02.py")

        # 4. 模型推理
        run_script("D:/Pythonproject/task04/final\optimization/inference.py")

        print("所有脚本运行完成！")
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
