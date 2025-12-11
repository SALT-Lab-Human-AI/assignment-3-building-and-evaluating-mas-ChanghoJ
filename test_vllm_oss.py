from openai import OpenAI

client = OpenAI(
    base_url="", # pragma: allowlist secret
    api_key="" # pragma: allowlist secret
)

result = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what MXFP4 quantization is."}
    ]
)

print(result.choices[0].message.content)

response = client.responses.create(
    model="openai/gpt-oss-20b",
    instructions="You are a helfpul assistant.",
    input="Explain what MXFP4 quantization is."
)

print(response.output_text)
