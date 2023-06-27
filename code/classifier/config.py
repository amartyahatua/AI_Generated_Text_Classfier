data_source = '../../../data/chatgpt_generated_wiki_data_1_5000.csv'

# feature config:
FEATURE_DIMENSION = 512
EXTRACTED_FEATURES = 1024

# Model paths
RF_MODLE_PATH = 'model/random_forest.pkl'
SVM_MODLE_PATH = 'model/svm.pkl'
XGB_MODLE_PATH = 'model/xgb.pkl'

# train_config:
BATCH_SIZE = 2
HIDDEN_LAYER = 32
LEAK = 0.2
MAP_DIM = 16
SINGLE_LAYER = 1
EPOCHS = 2
BETA1 = 0.05
BETA2 = 0.005
SAMPLE_SIZE = 100
SAMPLE_STEP = 50
SOFT_LABEL = 0.5
LR = 0.001