import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score
import spacy_universal_sentence_encoder
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from AI_Generated_Text_Classfier.code.classifier import config
from AI_Generated_Text_Classfier.code.classifier.helpers import *
from AI_Generated_Text_Classfier.code.classifier.dl_models.blstm import BiLSTMClassifier


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


class DL_Classifier:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_data()

        self.X_train = encode_text(self.X_train)
        self.X_test = encode_text(self.X_test)

        self.X_train = torch.tensor(self.X_train).float()
        self.y_train = torch.tensor(self.y_train.values).float()

        self.X_test = torch.tensor(self.X_test).float()
        self.y_test = torch.tensor(self.y_test.values).float()

        train = TensorDataset(self.X_train, self.y_train)
        test = TensorDataset(self.X_test, self.y_test)

        self.train_loader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=False)
        self.test_loader = DataLoader(test, batch_size=config.BATCH_SIZE, shuffle=False)

    def train(self):
        bilstm_model = BiLSTMClassifier(embedding_dim=config.FEATURE_DIMENSION, hidden_dim=config.HIDDEN_LAYER,
                                        label_size=1, batch_size=config.BATCH_SIZE)
        bilstm_model = bilstm_model.double()

        loss = nn.L1Loss()
        u_dis_optima = optim.Adam(bilstm_model.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
        for epoch in range(config.EPOCHS):
            total_loss = 0
            y_target_train = []
            y_predict_train = []
            for bi, (X_train, y_train) in enumerate(self.train_loader):
                X_train = torch.tensor(X_train).double()
                X_train = torch.nn.functional.normalize(X_train, p=2.0, dim=1)
                bilstm_model.zero_grad()
                lstm_r = bilstm_model(X_train.double())
                y_train = Variable(y_train.float())
                y_train = y_train.reshape(-1, 1)
                lstm_r_loss = loss(lstm_r.float(), y_train.float())
                total_loss += lstm_r_loss.item()
                lstm_r_loss.backward()
                u_dis_optima.step()

            avg_loss = total_loss / bi
            print('Epoch = {} and average loss = {}'.format(str(epoch), str(avg_loss)))

            y_pred = bilstm_model(self.X_test.double())
            y_pred = np.where(y_pred <= 0.5, 0, 1)
            f1 = f1_score(self.y_test, y_pred)
            print('F1 Score = ', f1)
            print(classification_report(self.y_test, y_pred))


if __name__ == '__main__':
    classifier_obj = DL_Classifier()
    classifier_obj.train()
