import pandas as pd
from sklearn.model_selection import train_test_split
from AI_Generated_Text_Classfier.code.classifier.config import *

# Load data and split into train and test datasets
def load_data():
    data = pd.DataFrame()
    df = pd.read_csv(data_source)
    data['text'] = pd.concat((df['Text'], df['GPT_Generated_Text']), axis=0)
    data['label'] = pd.concat(
        (pd.DataFrame([0] * df['Text'].shape[0]), pd.DataFrame([1] * df['GPT_Generated_Text'].shape[0])), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test
