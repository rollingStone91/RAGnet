# python -m vllm.entrypoints.openai.api_server \
#     --model ./models/Qwen3-4B-Instruct-2507 \
#     --tensor-parallel-size 4 \
#     --port 8000 \
#     --api-key my_secret_key \
#     --gpu-memory-utilization 0.5

from openai import OpenAI

# 指向本地 vLLM 服务
client = OpenAI(base_url="http://localhost:8000/v1", api_key="my_secret_key")

response = client.chat.completions.create(
    model="Qwen3-4B-Instruct-2507",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "介绍一下你自己"}
    ],
    temperature=0.7,
    max_tokens=128
)

print(response.choices[0].message.content)