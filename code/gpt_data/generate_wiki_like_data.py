API_KEY = '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'
import pandas as pd
import openai
from tqdm import tqdm
openai.api_key = API_KEY

df = pd.read_csv('../../data/wiki_data_extract.csv')
df['GPT_Generated_Text'] = ''*df.shape[0]

for i in tqdm(range(df.shape[0])):
    question_string = 'Describe {} {}'.format(df.iloc[i]['Page title'], df.iloc[i]['Section title'])
    print(question_string)
    try:
        chatcompletion_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": question_string}
            ])
        df.loc[i, 'GPT_Generated_Text'] = chatcompletion_response.to_dict()['choices'][0].to_dict()['message']['content']
    except:
        print('Error')
df.to_csv('../../data/chatgpt_generated_wiki_data.csv', index=False)