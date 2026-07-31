"""
目标：测试基线模型预测服务api接口

步骤：
1. 导包
2. 定义url，请求参数
3. 调用api接口
4. 打印结果
"""
# 1. 导包
import requests
import time

from sklearn.externals.array_api_compat.torch import result_type

# 2. 定义url，请求参数
url = "http://localhost:8000/predict"
# input_data = {"text":"这家餐厅的菜太好吃了，服务也很棒，强烈推荐！"}
input_data = {"text":"This restaurant is terrible. The food is cold and the service is extremely slow."}
# input_data = {"text":"Oh, great! 如此精美的包装，里面的东西却像 rubbish 一样。这真是我买过的最 stupid 的产品了。"}
# 3. 调用api接口
previous_time = time.time()
result = requests.post(url=url,json=input_data)
print(f"调用端时间:{time.time()-previous_time}")
# # 4. 打印结果
print(result.json())

# print(result)
# print(result.__repr__())
# print(result.__str__())
# print(str(result))