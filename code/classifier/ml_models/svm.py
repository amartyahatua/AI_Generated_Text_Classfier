import pickle
from sklearn import svm
from sklearn.metrics import f1_score
import spacy_universal_sentence_encoder
from sklearn.metrics import classification_report
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


class SVM_Classifier:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_data()

        self.X_train = encode_text(self.X_train)
        self.X_test = encode_text(self.X_test)

    def train(self):
        svm_clf = svm.SVC(C=4.0, kernel='rbf', degree=7, gamma='scale', coef0=0.0, shrinking=True, \
                         probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, \
                         max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)

        print("Model fitting started ....")
        model = svm_clf.fit(self.X_train, self.y_train)
        pickle.dump(model, open(config.SVM_MODLE_PATH, 'wb'))
        print('Model saved')
        y_pred = svm_clf.predict(self.X_test)
        y_pred = np.where(y_pred <= 0.5, 0, 1)
        f1 = f1_score(self.y_test, y_pred)
        print('F1 Score = ', f1)
        print(classification_report(self.y_test, y_pred))


if __name__ == '__main__':
    classifier_obj = SVM_Classifier()
    classifier_obj.train()
