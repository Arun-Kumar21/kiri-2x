import torch
from torch import nn

from config.config import CONFIG

class ModelTrainer:
    def __init__(self, model, device=CONFIG.DEVICE, learning_rate=CONFIG.LEARNING_RATE, epochs=CONFIG.EPOCHS):
        """
        Initialize the trainer with model, loss function, optimizer, and training parameters.
        :param model: The model to train.
        :param device: Device to run the training (CPU/GPU).
        :param learning_rate: Learning rate for optimization.
        :param epochs: Number of training epochs.
        """
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.epochs = epochs
    
    def train(self, train_loader, save_path="weights/srcnn_rgb.pth"):
        """
        Train the model.
        :param train_loader: DataLoader for training data.
        :param save_path: Path to save the trained model.
        """
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for lr, hr in train_loader:
                lr, hr = lr.to(self.device), hr.to(self.device)
                
                # Forward pass
                sr = self.model(lr)
                
                # Compute loss
                loss = self.criterion(sr, hr)
                total_loss += loss.item()
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Loss: {avg_loss:.3f} for epoch {epoch+1}")
        
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")