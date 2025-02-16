import torch
from torch import nn
import torch.nn.functional as F

from config.config import CONFIG

from data.data_loader import train_loader
from data.noise_data_loader import noise_train_loader

from models.vgg.vgg8_SR import VGG8_SR
from models.DnCNN.dncnn import DnCNN

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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), learning_rate)
        self.epochs = epochs
    
    def train(self, train_loader, save_path=None):
        """
        Train the model.
        :param train_loader: DataLoader for training data.
        :param save_path: Path to save the trained model.
        """
        try:
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

            if save_path: 
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            torch.save(self.model.state_dict(), "checkpoint.pth")
            print("Model saved. Exiting safely.")

if __name__ == "__main__":
    model = DnCNN()
    trainer = ModelTrainer(model)
    trainer.train(train_loader=noise_train_loader, save_path='weights/DnCNN_rgb.pth')