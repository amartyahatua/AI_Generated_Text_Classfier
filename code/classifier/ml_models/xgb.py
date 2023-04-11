import pickle
from sklearn.metrics import f1_score
import spacy_universal_sentence_encoder
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from AI_Generated_Text_Classfier.code.classifier import config
from AI_Generated_Text_Classfier.code.classifier.helpers import *


def encode_text(data):
    print('Encoding started')
    result = []
    nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
    for i in range(data.shape[0]):
        text = data.values.tolist()[i]
        result_temp = nlp(text[0])
        result.append(result_temp.vector)
    print('Encoding done')
    return result


class XGB_Classifier:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_data()

        self.X_train = encode_text(self.X_train)
        self.X_test = encode_text(self.X_test)

    def train(self):
        xgb_clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000)

        print("Model fitting started ....")
        model = xgb_clf.fit(self.X_train, self.y_train)
        pickle.dump(model, open(config.XGB_MODLE_PATH, 'wb'))
        print('Model saved')
        y_pred = xgb_clf.predict(self.X_test)
        y_pred = np.where(y_pred <= 0.5, 0, 1)
        f1 = f1_score(self.y_test, y_pred)
        print('F1 Score = ', f1)
        print(classification_report(self.y_test, y_pred))


if __name__ == '__main__':
    classifier_obj = XGB_Classifier()
    classifier_obj.train()
