API_KEY = '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'
import time
import pandas as pd
import openai
from tqdm import tqdm

openai.api_key = API_KEY

data = pd.read_csv('../../data/extracted_data_us_election_2024.csv')


def create_questions():
    for i in tqdm(range(data.shape[0])):
        if(i>1189):
            try:
                keywords = data.iloc[i]['Keywords']
                if keywords is  not None:
                    question_string = 'Give me 10 questions using the following key words: ' + keywords + '?'
                    chatcompletion_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                                           messages=[
                                                                               {"role": "system", "content": question_string}
                                                                           ])

                    question_list = chatcompletion_response.to_dict()['choices'][0].to_dict()['message']['content'].split('\n')

                    # Get question for every row and save in .csv file
                    temp = [data.iloc[i]['Title'], data.iloc[i]['Text'], data.iloc[i]['Summary'], keywords]

                    for i in range(len(question_list)):
                        if len(question_list[i].split('.')[-1].strip()):
                            temp.append(question_list[i].split('.')[-1].strip())
                    temp = pd.DataFrame([temp])
                    temp.to_csv('../../data/chatgpt_generated_us_election_2024_questions.csv', mode='a', index=False,
                                header=False)
            except Exception as e:
                print('Error message: ', e)
                time.sleep(30)
                i = i - 1


def generare_answers():
    data = pd.read_csv('../../data/chatgpt_generated_us_election_2024_questions.csv')
    for i in tqdm(range(data.shape[0])):
        temp_data = [data.iloc[i].to_list()[0], data.iloc[i].to_list()[1], data.iloc[i].to_list()[2], data.iloc[i].to_list()[3]]
        row_data = data.iloc[i].to_list()[4:-1]
        for count in range(len(row_data)):
            if row_data[count] is not None:
                print(row_data[count])
                question_string = row_data[count]
                try:
                    chatcompletion_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                                           messages=[
                                                                               {"role": "system",
                                                                                "content": question_string}
                                                                           ])

                    answer = chatcompletion_response.to_dict()['choices'][0].to_dict()['message']['content']
                    temp_data.append(answer)

                except Exception as e:
                    print('Error message: ', e)
                    time.sleep(30)
                #     count = count - 1
        temp_data = pd.DataFrame([temp_data])
        temp_data.to_csv('../../data/chatgpt_generated_us_election_2024_questions_answers.csv', mode='a', index=False, header=False)






#create_questions()
generare_answers()
