API_KEY = ''
import time
import pandas as pd
import openai
from tqdm import tqdm
openai.api_key = API_KEY

data = pd.read_csv('../../data/extracted_data_us_election_2024.csv')
keywords = data[' Keywords']
question_string = 'What are questions can be formed using all the key words: '+keywords+' ?'