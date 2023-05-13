API_KEY = ''
import time
import pandas as pd
import openai
from tqdm import tqdm

openai.api_key = API_KEY

data = pd.read_csv('../../data/extracted_data_us_election_2024.csv')


def create_questions():
    for i in tqdm(range(data.shape[0])):
        keywords = ', '.join(data.iloc[i]['Keywords'].tolist())
        question_string = 'What are questions can be formed using all the key words: ' + keywords + '?'
        try:
            chatcompletion_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": question_string}
                ])
            data.at[i, 'GPT_Generated_Text'] = chatcompletion_response.to_dict()['choices'][0].to_dict()['message']['content']
            data_individual = pd.DataFrame([data.iloc[i].values.transpose()])
            data_individual.to_csv('../../data/chatgpt_generated_us_election_2024_questions.csv', mode='a', index=False, header=False)
        except Exception as e:
            print('Error message: ', e)
            time.sleep(30)
            i = i-1
