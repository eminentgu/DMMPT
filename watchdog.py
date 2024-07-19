import subprocess
import time

while True:
    # 执行系统命令squeue
    result = subprocess.run(["squeue"], capture_output=True, text=True)
    time.sleep(5)
    # 检查返回结果中是否包含multi_prompt字段
    if "multi_prompt" not in result.stdout:
        # 记录当前时间
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # 执行sbatch命令
        subprocess.run(["sbatch", "-t", "0-24:0:0", "--mem=30G", "run.sh"])
        
        # 输出执行sbatch命令的时间
        print(f"执行sbatch命令的时间: {current_time}")
        print("已执行sbatch命令")
    
    # 等待10分钟
    time.sleep(600)  # 10分钟 = 10 * 60秒
