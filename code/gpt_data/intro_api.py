API_KEY = ''


import os
import openai
openai.api_key = API_KEY

prompt = 'What is Boston?'

response = openai.Completion.create(engine='text-davinci-001', prompt=prompt, max_tokens=6)
print(response)