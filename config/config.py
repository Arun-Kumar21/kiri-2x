import torch

class CONFIG:
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 100