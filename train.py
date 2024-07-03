import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_checkpoint(filepath):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        return checkpoint
    return None

def save_checkpoint(state, filepath):
    torch.save(state, filepath)

def train(checkpoint_file, max_time):
    start_time = time.time()
    
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model, criterion, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch = 0
    
    # Load the checkpoint if it exists
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch = checkpoint['epoch']
    
    model.train()
    while True:
        epoch += 1  #simulate indefinite training
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch} completed")

        # Save checkpoint periodically
        if epoch % 1 == 0:
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, checkpoint_file)

        # Check if max_time has been reached
        if time.time() - start_time > max_time:
            break
    
    # Save checkpoint at the end of the job
    save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, checkpoint_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Training script with checkpointing')
    parser.add_argument('--checkpoint_file', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--max_time', type=float, required=True, help='Maximum time to run the script in seconds')
    args = parser.parse_args()

    train(args.checkpoint_file, args.max_time)
