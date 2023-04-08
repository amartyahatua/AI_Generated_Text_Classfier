data_source = '../../data/chatgpt_generated_wiki_data_1_5000.csv'

# feature config:
FEATURE_DIMENSION = 512

# train_config:
BATCH_SIZE = 100
HIDDEN_LAYER = 32
LEAK = 0.2
MAP_DIM = 16
SINGLE_LAYER = 1
EPOCHS = 20
BETA1 = 0.05
BETA2 = 0.005
SAMPLE_SIZE = 100
SAMPLE_STEP = 50
SOFT_LABEL = 0.5
LR = 0.001