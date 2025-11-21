import json
import os
import dashscope
from dashscope import MultiModalConversation

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

messages = [{"role": "user", "content": [{"text": "一位安静的少女在观看远方的风景,达到杰作级别，展现出最佳摄影品质,展现古典美"}]}]
messages = [{"role": "user", "content":
             [{"text": "在一个安静的小镇，一位少女正坐在窗边阅读红楼梦，旁边有一只小花猫，窗外有小桥流水，房屋的装修风格古典肃静,达到杰作级别，展现出最佳摄影品质,展现古典美"}]}]
api_key = 'sk-08f9217e9e2d4c9d84a911c9976ced08'

response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-plus",
    messages=messages,
    result_format='message',
    stream=False,
    watermark=False,
    prompt_extend=True,
    negative_prompt='',
    size='1328*1328'
)

if response.status_code == 200:
    print(json.dumps(response, ensure_ascii=False))
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
