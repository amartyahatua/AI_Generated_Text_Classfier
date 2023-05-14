API_KEY = ''
import time
import pandas as pd
import openai
from tqdm import tqdm

openai.api_key = API_KEY

data = pd.read_csv('../../data/extracted_data_us_election_2024.csv')


def create_questions():
    for i in tqdm(range(data.shape[0])):
        keywords = data.iloc[i]['Keywords']
        question_string = 'What are questions can be formed using all the key words: ' + keywords + '?'
        try:
            chatcompletion_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                                   messages=[
                                                                       {"role": "system", "content": question_string}
                                                                   ])

            question_list = chatcompletion_response.to_dict()['choices'][0].to_dict()['message']['content'].split('\n')

            # Get question for every row and save in .csv file
            temp = [data.iloc[i]['Title'], data.iloc[i]['Text'], data.iloc[i]['Summary'], keywords]

            for i in range(len(question_list)):
                if (len(question_list[i].split('.')[-1].strip())):
                    temp.append(question_list[i].split('.')[-1].strip())
            temp = pd.DataFrame([temp])
            temp.to_csv('../../data/chatgpt_generated_us_election_2024_questions.csv', mode='a', index=False,
                        header=False)
        except Exception as e:
            print('Error message: ', e)
            time.sleep(30)
            i = i - 1


def generare_answers():
    pass



create_questions()
generare_answers()
