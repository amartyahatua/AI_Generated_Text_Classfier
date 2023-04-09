import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from AI_Generated_Text_Classfier.code.classifier.config import *


# Preprocessing and cleaning the data
def data_preprocessing(df):
    df.dropna(inplace=True)
    clean_data = []

    for x in df:
        new_text = re.sub('<.*?>', '', x)  # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text)  # remove punc
        new_text = re.sub('<unk>', '', new_text)
        new_text = re.sub(r'\d+', '', new_text)  # remove numbers
        new_text = re.sub(' +', ' ', new_text)  # remove extra space
        new_text = new_text.lstrip()  # remove leading whitespaces
        new_text = new_text.replace("\n", " ")
        new_text = new_text.lower()

        if new_text != '':
            clean_data.append(new_text)
    clean_data = pd.DataFrame(clean_data)
    return clean_data


# Load data and split into train and test datasets
def load_data():
    data = pd.DataFrame()
    df = pd.read_csv(data_source)
    df = df.iloc[0:10, :]

    df['Text'] = data_preprocessing(df['Text'])
    df['GPT_Generated_Text'] = data_preprocessing(df['GPT_Generated_Text'])

    data['text'] = pd.concat((df['Text'], df['GPT_Generated_Text']), axis=0)
    data['label'] = pd.concat((pd.DataFrame([0] * df['Text'].shape[0]), \
                               pd.DataFrame([1] * df['GPT_Generated_Text'].shape[0])), axis=0)

    data['text'].replace('', np.nan, inplace=True)
    data.dropna(subset=['text'], inplace=True)
    data["word_count"] = data["text"].apply(lambda x: len(x))
    data = data[(data["word_count"] >= 100)]
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test
