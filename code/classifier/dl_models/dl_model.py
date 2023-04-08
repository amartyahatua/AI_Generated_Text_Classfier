import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import spacy_universal_sentence_encoder
from torch.utils.data import DataLoader
from AI_Generated_Text_Classfier.code.classifier import config
from AI_Generated_Text_Classfier.code.classifier.helpers import *
from AI_Generated_Text_Classfier.code.classifier.dl_models.blstm import BiLSTMClassifier


def encode_text(data):
    print('Encoding started')
    result = []
    nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
    for text in data:
        result_temp = nlp(text)
        result.append(result_temp.vector)
    print('Encoding done')
    return result


class DL_Classifier:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_data()

        self.X_train = encode_text(self.X_train)
        self.X_train = DataLoader(self.X_train, config.BATCH_SIZE, shuffle=False)
        self.y_train = DataLoader(self.y_train, config.BATCH_SIZE, shuffle=False)

        self.X_test = encode_text(self.X_test)
        self.X_test = DataLoader(self.X_test, config.BATCH_SIZE, shuffle=False)
        self.y_test = DataLoader(self.y_test, config.BATCH_SIZE, shuffle=False)

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
            for bi, data in (enumerate(zip(self.X_train, self.y_train))):
                X_train, y_train = data
                X_train = torch.tensor(X_train).double()
                X_train = torch.nn.functional.normalize(X_train, p=2.0, dim=1)

                bilstm_model.zero_grad()
                lstm_r = bilstm_model(X_train.double())
                y_train = Variable(y_train.float())
                y_train = y_train.reshape(-1, 1)
                lstm_r_loss = loss(lstm_r.float(), y_train.float())
                total_loss += lstm_r_loss
                lstm_r_loss.backward()
                u_dis_optima.step()

                label = y_train.reshape(1, -1)
                label = label.data.tolist()[0]
                lstm_r = lstm_r.reshape(1, -1)
                outPred = np.where(lstm_r <= 0.5, 0, 1)
                y_target_train.extend(label)
                y_predict_train.extend(outPred.tolist()[0])


if __name__ == '__main__':
    classifier_obj = DL_Classifier()
    classifier_obj.train()