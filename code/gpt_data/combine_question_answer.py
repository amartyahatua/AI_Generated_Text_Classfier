import numpy as np
import scipy
import pandas as pd

questions = pd.read_csv('../../data/chatgpt_generated_us_election_2024_questions.csv')
answers = pd.read_csv('../../data/chatgpt_generated_us_election_2024_questions_answers.csv')

print(questions.shape)
print(questions.columns)

print(answers.shape)
print(answers.columns)

results = pd.concat([questions['Title'], questions['Text'], questions['Summary'], questions['Keywords'], questions['Question 1'], answers['Answer 1'],
                     questions['Question 2'], answers['Answer 2'],
                     questions['Question 3'], answers['Answer 3'],
                     questions['Question 4'], answers['Answer 4'],
                     questions['Question 5'], answers['Answer 5'],
                     questions['Question 6'], answers['Answer 6'],
                     questions['Question 7'], answers['Answer 7'],
                     questions['Question 8'], answers['Answer 8'],
                     questions['Question 9'], answers['Answer 9'],
                     questions['Question 10'], answers['Answer 10']], axis=1)
print(results.shape)
results.to_csv('../../data/chatgpt_generated_us_election_2024_questions_answers_combine.csv', index=False)