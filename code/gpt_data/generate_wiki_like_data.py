API_KEY = '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'
import time
import pandas as pd
import openai
from tqdm import tqdm
openai.api_key = API_KEY

df = pd.read_csv('../../data/wiki_data_extract_1_50000.csv')
df['GPT_Generated_Text'] = ''*df.shape[0]

result = pd.DataFrame()
for i in tqdm(range(df.shape[0])):
    if(i>20066):
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
            data_individual = pd.DataFrame([df.iloc[i].values.transpose()])
            data_individual.to_csv('../../data/chatgpt_generated_wiki_data_1_50000.csv', mode='a', index=False, header=False)
        except Exception as e:
            print('Error message: ', e)
            time.sleep(30)
            i = i-1




