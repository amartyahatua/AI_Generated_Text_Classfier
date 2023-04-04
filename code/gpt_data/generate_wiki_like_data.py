API_KEY = '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'
import pandas as pd
import openai
from tqdm import tqdm
openai.api_key = API_KEY

df = pd.read_csv('../../data/wiki_data_extract_1_5000.csv')
df['GPT_Generated_Text'] = ''*df.shape[0]

result = pd.DataFrame()
for i in tqdm(range(df.shape[0])):
    excluded_strings = ['See also', 'References', 'External links']
    if any(ext in df.iloc[i]['Section title'] for ext in excluded_strings):
        print(df.iloc[i]['Section title'])
        continue

    question_string = 'Describe {} {}'.format(df.iloc[i]['Page title'], df.iloc[i]['Section title'])
    try:
        chatcompletion_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": question_string}
            ])
        df.at[i, 'GPT_Generated_Text'] = chatcompletion_response.to_dict()['choices'][0].to_dict()['message']['content']
    except Exception as e:
        print('Error message: ', e)

df.to_csv('../../data/chatgpt_generated_wiki_data_1_5000.csv', index=False)

