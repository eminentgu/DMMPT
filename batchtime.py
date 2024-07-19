import re

# 打开文件并读取内容
with open('logs.txt', 'r') as file:
    content = file.read()

# 定义正则表达式模式，匹配浮点数
# 这个模式匹配一个或多个数字，接着是一个点，后面再跟一个或多个数字
pattern = r'time:([0-9]+\.[0-9]+)'

# 使用findall方法查找所有匹配项
times = re.findall(pattern, content)

# 将匹配到的时间字符串转换为浮点数，然后转换为整数（截断小数部分）
times = [float(time) for time in times if float(time) >= 1]

print(times)

print(sum(times)/len(times)) #batch1    1.3s   1.3/32*100 = 4.0625

