import shap
import pickle
from sklearn.metrics import f1_score
import spacy_universal_sentence_encoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from AI_Generated_Text_Classfier.code.classifier import config
from AI_Generated_Text_Classfier.code.classifier.helpers import *
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


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
        gt_features = pd.read_csv('../../../../data/features_GT.csv')
        gpt_features = pd.read_csv('../../../../data/features_ChatGPT.csv')
        print(gpt_features.shape)
        gt_features = gt_features.iloc[:,1:1025]
        gpt_features = gpt_features.iloc[:,1:1025]

        gt_class = pd.DataFrame(np.zeros(gt_features.shape[0]))  # Ground Truth --> 0
        gpt_class = pd.DataFrame(np.ones(gpt_features.shape[0]))  # GPT --> 1

        X = pd.concat([gt_features, gpt_features], axis=0)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0)
        print('Input total shape:', X.shape)


        y = pd.concat([gt_class, gpt_class], axis=0)
        y.fillna(0)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33, random_state=42)

    def train(self):
        xgb_clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000)
        print("Model fitting started ....")
        model = xgb_clf.fit(self.X_train, self.y_train)
        pickle.dump(model, open(config.XGB_MODLE_PATH, 'wb'))
        print('Model saved')
        y_pred = xgb_clf.predict(self.X_test)
        # y_pred = np.where(y_pred <= 0.5, 0, 1)
        f1 = f1_score(self.y_test, y_pred)
        print('F1 Score = ', f1)
        print(classification_report(self.y_test, y_pred))

        # explainer = shap.Explainer(model)
        # self.X_train = pd.DataFrame(self.X_train)
        # shap_values = explainer(self.X_train)
        #
        # # visualize the first prediction's explanation
        # shap.plots.waterfall(shap_values[0])

    def test(self):
        print(self.X_test.shape)
        print(self.y_test.shape)
        self.X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.X_test.fillna(0)
        print(self.X_test.shape)

        # loaded_model = pickle.load(open(config.XGB_MODLE_PATH, 'rb'))
        # predicted_class = loaded_model.predict(self.X_test)


if __name__ == '__main__':
    classifier_obj = XGB_Classifier()
    classifier_obj.train()
    #classifier_obj.test()
