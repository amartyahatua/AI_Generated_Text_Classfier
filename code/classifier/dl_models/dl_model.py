import torch
import torch.nn as nn
import torch.optim as optim
from AI_Generated_Text_Classfier.code.classifier import config
from AI_Generated_Text_Classfier.code.classifier.dl_models.blstm import BiLSTMClassifier



class DL_Classifier:

    def train(self):
        bilstm_model = BiLSTMClassifier(embedding_dim=config.FEATURE_DIMENSION, hidden_dim=config.HIDDEN_LAYER,
                                       label_size=1, batch_size=config.BATCH_SIZE)
        bilstm_model = bilstm_model.double()

        loss = nn.L1Loss()
        u_dis_optim = optim.Adam(bilstm_model.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
        for epoch in range(config.EPOCHS):
            total_loss = 0
            count = 0
            y_target_train = []
            y_predict_train = []
            for bi, data in (enumerate(zip(self.ground_truth, self.label))):
                ground_truth, label = data
                ground_truth = torch.tensor(ground_truth).double()
                ground_truth = torch.nn.functional.normalize(ground_truth, p=2.0, dim=1)