from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


models = client.models.list()
model = models.data[0].id
print('model', model)
chat_response = client.chat.completions.create(
    #model="Qwen/Qwen3-0.6B",
    #model = 'facebook_opt-125m',
    model = model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "讲一个笑话"},
        #{"role": "user", "content": "Tell me a joke."},
    ],
    #extra_body={'chat_template':''}
)
print("Chat response:", chat_response)
