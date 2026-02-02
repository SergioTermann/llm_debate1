from openai import OpenAI

api_key = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"
client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

print("Testing API connection...")
try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[{"role": "user", "content": "Say 'Hello'"}],
        temperature=0.7,
        timeout=30.0
    )
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
