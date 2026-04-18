from openai import OpenAI

# Use the URL printed in Kaggle
client = OpenAI(
    base_url="https://<your-localtunnel-url>.loca.lt/v1", 
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local",
    temperature=0.1,
    messages=[
        {"role": "system", "content": "You are Pocket-Agent..."},
        {"role": "user", "content": "Book me a flight to Hawaii"}
    ]
)

print(response.choices[0].message.content)
