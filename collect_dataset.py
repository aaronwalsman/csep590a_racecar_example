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

import PIL.Image as Image

def collect_dataset(
    name,
    num_examples,
    resolution,
    seed,
    examples,
    gui,
    max_steering,
    steering_noise,
    speed,
    speed_noise,
    policy=None,
):
    env = gym.wrappers.AutoResetWrapper(
        gym.make(
            'Example-Racetrack-v0',
            max_steering=max_steering,
            steering_noise=steering_noise,
            speed=speed,
            speed_noise=speed_noise,
            gui=gui,
        )
    )
    o,i = env.reset(seed=seed)
    images = np.zeros(
        (num_examples, resolution, 4*resolution, 3), dtype=np.uint8)
    waypoints = np.zeros((num_examples,), dtype=int)
    actions = np.zeros((num_examples,), dtype=float)
    for step in tqdm.tqdm(range(num_examples)):
        if policy is None:
            action = o['expert']
        else:
            x = torch.Tensor(o['image']).to('cuda').float() / 255.
            x = x * 2. - 1.
            x = x.permute(2,0,1)
            c,h,w = x.shape
            x = x.view(1,c,h,w)
            w = torch.LongTensor([o['waypoint']]).to('cuda')
            action = policy(x,w)[0].detach().cpu().numpy()
        o,r,t,u,i = env.step(action)
        images[step] = o['image']
        waypoints[step] = o['waypoint']
        actions[step] = o['expert']
        
        if step < examples:
            Image.fromarray(images[step]).save('example_%06i.png'%step)
    
    if name is not None:
        np.save('%s_images.npy'%name, images)
        np.save('%s_waypoints.npy'%name, waypoints)
        np.save('%s_actions.npy'%name, actions)
    
    return images, waypoints, actions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--examples', type=int, default=0)
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
    collect_dataset(
        args.split,
        args.size,
        args.resolution,
        args.seed,
        args.examples,
        args.gui,
        args.max_steering,
        args.steering_noise,
        args.speed,
        args.speed_noise,
    )
