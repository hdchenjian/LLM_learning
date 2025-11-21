from openai import OpenAI
import dashscope
from http import HTTPStatus
import numpy as np

def test_embedding(text, api_key):
    resp = dashscope.TextEmbedding.call(
        api_key=api_key,
        model="text-embedding-v4",
        #model='text-embedding-v3',
        input=text,
        dimension=1024,
        output_type="dense&sparse"
    )
    if resp.status_code == HTTPStatus.OK:
        #print(type(resp), type(resp.get('output')))
        return resp.get('output')['embeddings'][0]['embedding']
    return

def get_response(client, messages):
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=messages
    )
    print('get_response', len(completion.choices[0].message.content), completion.choices[0].message.role)
    return (completion.choices[0].message.content, completion.choices[0].message.role)

def test_chat(api_key):
    client = OpenAI(api_key= api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
    messages = []
    messages.append({"role": "user", "content": "推荐一部关于太空探索的科幻电影。"})
    print("第1轮")
    print(f"用户：{messages[0]['content']}")
    assistant_output = get_response(client, messages)[0]
    messages.append({"role": "assistant", "content": assistant_output})
    print(f"模型：{assistant_output}\n")

    # 第 2 轮
    messages.append({"role": "user", "content": "这部电影的导演是谁？"})
    print("第2轮")
    print(f"用户：{messages[-1]['content']}")
    assistant_output = get_response(client, messages)[0]
    messages.append({"role": "assistant", "content": assistant_output})
    print(f"模型：{assistant_output}\n")

def test_qwen(api_key):
    client = OpenAI(api_key= api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    completion = client.chat.completions.create(
        model="qwen3-max", 
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            #{'role': 'user', 'content': '简单介绍一下linux'}
            #{'role': 'user', 'content': '你是谁?'}
            {'role': 'user', 'content': """今天真开心。-->正向 心情不太好。-->负向 我们是快乐的年轻人。-->"""}
        ],
        temperature=1,
        top_p=1,
    )
    print(completion.choices[0].message.content)

def test_openai():
    #proxy_url = 'socks5://127.0.0.1'
    #proxy_port = '1080'
    #os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
    #os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'
    
    OPENAI_API_KEY = \
        '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'
    client = OpenAI(api_key=OPENAI_API_KEY)
    text = "我喜欢你"
    model = "text-embedding-ada-002"
    emb_req = client.embeddings.create(input=[text], model=model)
    emb = emb_req.data[0].embedding
    len(emb), type(emb)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_embedding():
    api_key = 'sk-08f9217e9e2d4c9d84a911c9976ced08'
    embedding_1 = test_embedding('我喜欢你', api_key)
    print(type(embedding_1))
    embedding_2 = test_embedding('我钟意你', api_key)
    embedding_3 = test_embedding('我恨你', api_key)
    print('embedding length', len(embedding_1), len(embedding_2), len(embedding_3))
    print(cosine_similarity(embedding_1, embedding_2))
    print(cosine_similarity(embedding_1, embedding_3))
    print(cosine_similarity(embedding_2, embedding_3))

if __name__ == '__main__':
    api_key = 'sk-08f9217e9e2d4c9d84a911c9976ced08'
    #test_qwen(api_key)
    test_chat(api_key)
    
