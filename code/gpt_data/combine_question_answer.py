import numpy as np
import scipy
import pandas as pd

questions = pd.read_csv('../../data/chatgpt_generated_us_election_2024_questions.csv')
answers = pd.read_csv('../../data/chatgpt_generated_us_election_2024_questions_answers.csv')

print(questions.shape)
print(questions.columns)

print(answers.shape)
print(answers.columns)
questions = questions.replace(np.nan, '', regex=True)
answers = answers.replace(np.nan, '', regex=True)
combine_answer = pd.DataFrame()

for i in range(answers.shape[0]):
    temp_answer = []
    temp_answer.extend(answers['Answer 1'])
    temp_answer.extend(answers['Answer 2'])
    temp_answer.extend(answers['Answer 3'])
    temp_answer.extend(answers['Answer 4'])
    temp_answer.extend(answers['Answer 5'])
    temp_answer.extend(answers['Answer 6'])
    temp_answer.extend(answers['Answer 7'])
    temp_answer.extend(answers['Answer 8'])
    temp_answer.extend(answers['Answer 9'])
    temp_answer.extend(answers['Answer 10'])
    temp_answer = pd.DataFrame([[temp_answer]], columns=['Combined answer'])
    combine_answer = pd.concat([combine_answer, temp_answer], axis=0)



combine_answer = combine_answer.reset_index(drop=True)
results = pd.concat([questions['Title'], questions['Text'], questions['Summary'], questions['Keywords'], questions['Question 1'], answers['Answer 1'],
                     questions['Question 2'], answers['Answer 2'],
                     questions['Question 3'], answers['Answer 3'],
                     questions['Question 4'], answers['Answer 4'],
                     questions['Question 5'], answers['Answer 5'],
                     questions['Question 6'], answers['Answer 6'],
                     questions['Question 7'], answers['Answer 7'],
                     questions['Question 8'], answers['Answer 8'],
                     questions['Question 9'], answers['Answer 9'],
                     questions['Question 10'], answers['Answer 10'], combine_answer], axis=1)
print(results.shape)
results.to_csv('../../data/chatgpt_generated_us_election_2024_questions_answers_combine.csv', index=False)
print(results.shape)
