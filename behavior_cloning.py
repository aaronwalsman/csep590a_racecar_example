import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import tqdm

from model import DrivingModel

class DrivingDataset:
    def __init__(self, images, waypoints, actions):
        self.images = images
        self.waypoints = waypoints
        self.actions = actions
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        return self.images[i], self.waypoints[i], self.actions[i]

def behavior_cloning(
    epochs = 5,
    batch_size=256,
    learning_rate = 3e-4,
):
    train_images = np.load('train_images.npy')
    train_waypoints = np.load('train_waypoints.npy')
    train_actions = np.load('train_actions.npy')
    train_dataset = DrivingDataset(
        train_images, train_waypoints, train_actions)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    
    test_images = np.load('train_images.npy')
    test_waypoints = np.load('train_waypoints.npy')
    test_actions = np.load('train_actions.npy')
    test_dataset = DrivingDataset(
        test_images, test_waypoints, test_actions)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    
    model = DrivingModel().to('cuda')
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    train_epochs(epochs, model, optimizer, train_loader, test_loader)
    torch.save(model.state_dict(), 'bc.pt')

def train_epochs(epochs, model, optimizer, train_loader, test_loader):
    all_losses = []
    running_loss = None
    for epoch in range(1, epochs+1):
        print('Epoch: %i'%epoch)
        iterate = tqdm.tqdm(train_loader)
        for x,w,a in iterate:
            x = torch.Tensor(x).to('cuda').float() / 255.
            x = x * 2. - 1.
            x = x.permute(0,3,1,2)
            w = torch.LongTensor(w).to('cuda')
            a_hat = model(x,w)
            a = torch.Tensor(a).to('cuda').float()
            loss = torch.nn.functional.huber_loss(a_hat, a)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            all_losses.append(float(loss))
            if running_loss is None:
                running_loss = float(loss)
            else:
                running_loss = running_loss * 0.9 + float(loss) * 0.1
            iterate.set_description('Training Loss: %.04f'%running_loss)
        
        if test_loader is not None:
            with torch.no_grad():
                iterate = tqdm.tqdm(test_loader)
                iterate.set_description('Testing')
                errors = []
                for x,w,a in iterate:
                    x = torch.Tensor(x).to('cuda').float() / 255.
                    x = x * 2. - 1.
                    x = x.permute(0,3,1,2)
                    w = torch.LongTensor(w).to('cuda')
                    a_hat = model(x,w)
                    a = torch.Tensor(a).to('cuda').float()
                    errors.append(torch.abs(a - a_hat))
                error = torch.cat(errors).mean()
                print('Average Test Error: %f'%error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    
    args = parser.parse_args()
    behavior_cloning(
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
