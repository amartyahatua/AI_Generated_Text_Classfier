API_KEY = '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'


import os
import openai
openai.api_key = API_KEY


# Testing two APIs
prompt = 'What is Boston?'

completion_response = openai.Completion.create(engine='text-davinci-001', prompt=prompt, max_tokens=6)
print(completion_response)

chatcompletion_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "What is Python Programing?"}
    ])
print(chatcompletion_response)