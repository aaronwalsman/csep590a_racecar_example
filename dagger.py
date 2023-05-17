import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm

from model import DrivingModel
from behavior_cloning import DrivingDataset, train_epochs
from collect_dataset import collect_dataset
from racetrack import (
    DEFAULT_MAX_STEERING,
    DEFAULT_STEERING_NOISE,
    DEFAULT_SPEED,
    DEFAULT_SPEED_NOISE,
)

def dagger(
    rounds=5,
    steps_per_round=10000,
    epochs=5,
    batch_size=256,
    learning_rate=3e-4,
    resolution=32,
    max_steering=DEFAULT_MAX_STEERING,
    steering_noise=DEFAULT_STEERING_NOISE,
    speed=DEFAULT_SPEED,
    speed_noise=DEFAULT_SPEED_NOISE,
    gui=False,
):
    model = None
    round_images = []
    round_waypoints = []
    round_actions = []
    for r in range(1, rounds+1):
        print('Round: %i'%r)
        images, waypoints, actions = collect_dataset(
            None,
            steps_per_round,
            resolution,
            None,
            0,
            gui,
            max_steering,
            steering_noise,
            speed,
            speed_noise,
            policy=model,
        )
        round_images.append(images)
        round_waypoints.append(waypoints)
        round_actions.append(actions)
        all_images = np.concatenate(round_images)
        all_waypoints = np.concatenate(round_waypoints)
        all_actions = np.concatenate(round_actions)
        train_dataset = DrivingDataset(all_images, all_waypoints, all_actions)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        
        model = DrivingModel().to('cuda')
        optimizer = Adam(model.parameters(), lr=learning_rate)
        train_epochs(epochs, model, optimizer, train_loader, None)
    
    torch.save(model.state_dict(), 'dg.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--steps-per-round', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gui', action='store_true')

    args = parser.parse_args()
    dagger(
        args.rounds,
        args.steps_per_round,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        gui=args.gui,
    )
