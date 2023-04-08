import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.DataFrame()
    df = pd.read_csv('../../data/chatgpt_generated_wiki_data_1_5000.csv')
    data['text'] = pd.concat((df['Text'],df['GPT_Generated_Text']),axis=0)
    data['label'] = pd.concat((pd.DataFrame([0]*df['Text'].shape[0]),pd.DataFrame([1]*df['GPT_Generated_Text'].shape[0])),axis=0)



load_data()