import pickle
from sklearn.metrics import f1_score
import spacy_universal_sentence_encoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
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


class RF_Classifier:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_data()

        self.X_train = encode_text(self.X_train)
        self.X_test = encode_text(self.X_test)

    def train(self):
        rf_clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, \
                                        min_samples_split=10, min_samples_leaf=10, min_weight_fraction_leaf=0.0, \
                                        max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, \
                                        bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, \
                                        warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

        print("Model fitting started ....")
        model = rf_clf.fit(self.X_train, self.y_train)
        pickle.dump(model, open(config.RF_MODLE_PATH, 'wb'))
        print('Model saved')
        y_pred = rf_clf.predict(self.X_test)
        y_pred = np.where(y_pred <= 0.5, 0, 1)
        f1 = f1_score(self.y_test, y_pred)
        print('F1 Score = ', f1)
        print(classification_report(self.y_test, y_pred))


if __name__ == '__main__':
    classifier_obj = RF_Classifier()
    classifier_obj.train()
