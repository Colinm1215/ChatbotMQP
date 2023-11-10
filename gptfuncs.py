from openai import OpenAI
import json

client = OpenAI(api_key = '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')
functions = {}

def Chat(text):

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": text}
    ]
    )
    return response.choices[0].message.content
