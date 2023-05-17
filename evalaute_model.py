import argparse

import numpy as np

import torch

import tqdm

import gymnasium as gym

from racetrack import (
    register_racetrack,
    DEFAULT_MAX_STEERING,
    DEFAULT_STEERING_NOISE,
    DEFAULT_SPEED,
    DEFAULT_SPEED_NOISE,
)
register_racetrack()

from model import DrivingModel

def evaluate_model(
    model,
    steps,
    seed,
    gui,
    max_steering,
    steering_noise,
    speed,
    speed_noise
):
    if isinstance(model, str):
        model_dict = torch.load(model)
        model = DrivingModel().to('cuda')
        model.load_state_dict(model_dict)
    
    env = gym.wrappers.AutoResetWrapper(
        gym.make(
            'Example-Racetrack-v0',
            gui=gui,
            max_steering=max_steering,
            steering_noise=steering_noise,
            speed=speed,
            speed_noise=speed_noise,
        )
    )
    o,i = env.reset(seed=seed)
    for step in tqdm.tqdm(range(steps)):
        x = torch.Tensor(o['image']).to('cuda').float() / 255.
        x = x * 2. - 1.
        x = x.permute(2,0,1)
        c,h,w = x.shape
        x = x.view(1,c,h,w)
        w = torch.LongTensor([o['waypoint']]).to('cuda')
        action = model(x,w)[0].detach().cpu().numpy()
        o,r,t,u,i = env.step(action)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bc.pt')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument(
        '--max-steering', type=float, default=DEFAULT_MAX_STEERING)
    parser.add_argument(
        '--steering-noise', type=float, default=DEFAULT_STEERING_NOISE)
    parser.add_argument(
        '--speed', type=float, default=DEFAULT_SPEED)
    parser.add_argument(
        '--speed-noise', type=float, default=DEFAULT_SPEED_NOISE)
    
    args = parser.parse_args()
    evaluate_model(
        args.model,
        args.steps,
        args.seed,
        args.gui,
        args.max_steering,
        args.steering_noise,
        args.speed,
        args.speed_noise,
    )
