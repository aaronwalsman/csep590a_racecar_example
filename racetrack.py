import math

import numpy as np

import torch

import gymnasium as gym

import tqdm

import pybullet as p
import pybullet_data

DEFAULT_MAX_STEERING = math.pi
DEFAULT_STEERING_NOISE = 0.0
DEFAULT_SPEED = 0.05
DEFAULT_SPEED_NOISE = 0.0

class Racetrack(gym.Env):
    def __init__(
        self,
        max_steering=DEFAULT_MAX_STEERING,
        steering_noise=DEFAULT_STEERING_NOISE,
        speed=DEFAULT_SPEED,
        speed_noise=DEFAULT_SPEED_NOISE,
        resolution=32,
        gui=False,
    ):
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        
        self.create_scene()
        self.add_robot()
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1]), high=np.array([1]))
        self.observation_space = gym.spaces.Dict({
            'image' : gym.spaces.Box(
                low=0,
                high=255,
                shape=(resolution,resolution*4,3),
                dtype=np.uint8
            ),
            'waypoint':gym.spaces.Discrete(len(self.waypoints)),
            'expert':self.action_space,
        })
        
        self.x = [0,0,0]
        self.max_steering = max_steering
        self.steering_noise = steering_noise
        self.speed = speed
        self.speed_noise = speed_noise
        self.resolution = resolution
    
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.x = [0,0,0]
        self.move_car()
        self.current_waypoint = 0
        o = self.get_observation()
        
        self.update_waypoint_drawing()
        
        return o, {}
    
    def step(self, action):
        # get steering angle
        angle = action * self.max_steering
        angle = angle + self.np_random.uniform(
            -self.steering_noise, self.steering_noise
        )
        distance = self.speed + self.np_random.uniform(
            -self.speed_noise, self.speed_noise
        )
        self.x[2] += angle
        self.x[0] += np.cos(self.x[2]) * distance
        self.x[1] += np.sin(self.x[2]) * distance
        self.move_car()
        
        closest_waypoint, distance = self.get_closest_waypoint()
        next_waypoint = (self.current_waypoint+1)%len(self.waypoints)
        if distance > 1.:
            t = True
            r = -1
        else:
            t = False
            if closest_waypoint == next_waypoint:
                r = 1
                self.current_waypoint = next_waypoint
            else:
                r = 0
        
        self.update_waypoint_drawing()
        
        return self.get_observation(),r,t,False,{}
    
    def get_observation(self):
        image = self.render_panorama()
        expert = self.get_expert_advice()
        return {
            'image' : image,
            'waypoint' : self.current_waypoint,
            'expert' : expert,
        }
    
    def get_expert_advice(self):
        prev_waypoint = (self.current_waypoint - 1) % len(self.waypoints)
        prev_waypoint_position = self.waypoints[prev_waypoint][:2]
        current_waypoint_position = self.waypoints[self.current_waypoint][:2]
        next_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
        next_waypoint_position = self.waypoints[next_waypoint][:2]
        position = self.x[:2]
        
        distance_to_prev = np.linalg.norm(prev_waypoint_position - position)
        distance_to_next = np.linalg.norm(next_waypoint_position - position)
        if distance_to_prev < distance_to_next:
            v1 = current_waypoint_position - prev_waypoint_position
            v2 = position - prev_waypoint_position
            t = np.dot(v1,v2) / np.dot(v1,v1)
            target = (
                current_waypoint_position * (1. - t) +
                next_waypoint_position * t
            )
        else:
            next_next_waypoint = (
                (self.current_waypoint + 2) % len(self.waypoints))
            next_next_waypoint_position = self.waypoints[next_next_waypoint][:2]
            v1 = next_waypoint_position - current_waypoint_position
            v2 = position - current_waypoint_position
            t = np.dot(v1,v2) / np.dot(v1,v1)
            target = (
                next_waypoint_position * (1. - t) +
                next_next_waypoint_position * t
            )
        
        #next_waypoint_position = self.waypoints[next_waypoint]
        #offset = next_waypoint_position[:2] - self.x[:2]
        offset = target - position
        offset = offset / np.linalg.norm(offset)
        angle = np.arctan2(offset[1], offset[0])
        angle = self.minimized_angle(angle - self.x[2])
        angle = np.clip(angle, -self.max_steering, self.max_steering)
        angle = angle / self.max_steering
        return float(angle)
    
    def create_scene(self):
        self.plane_id = p.loadURDF('plane.urdf')
        
        self.waypoints = np.array([
            [ 0, 0, 0],
            [ 1, 0, 0],
            [ 2, 0, 0],
            [ 3, 0, 0],
            [ 4, 0, 0],
            [ 4, 1, 0],
            [ 4, 2, 0],
            [ 4, 3, 0],
            [ 3, 3, 0],
            [ 2, 3, 0],
            [ 1, 3, 0],
            [ 0, 3, 0],
            [-1, 3, 0],
            [-2, 3, 0],
            [-3, 3, 0],
            [-3, 2, 0],
            [-3, 1, 0],
            [-3, 0, 0],
            [-2, 0, 0],
            [-1, 0, 0],
        ])
        color = np.zeros((len(self.waypoints), 3))
        self.waypoint_id = p.addUserDebugPoints(
            self.waypoints, color, pointSize=10)
        
        # inside wall
        wall_id = p.loadURDF('inside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-2, 1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('inside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-1, 1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('inside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [0, 1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('inside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [1, 1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('inside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [2, 1, 0], [0,0,0,1])
        
        #=====
        
        wall_id = p.loadURDF('inside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [3, 1, 0], [0,0,0.707,0.707])
        
        #=====
        
        wall_id = p.loadURDF('inside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-2, 2, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('inside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-1, 2, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('inside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [0, 2, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('inside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [1, 2, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('inside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [2, 2, 0], [0,0,0,1])
        
        #=====
        
        wall_id = p.loadURDF('inside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-2, 1, 0], [0,0,0.707,0.707])
        
        #=====
        
        # outside wall
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-4, -1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-3, -1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-2, -1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-1, -1, 0], [0,0,0,1])

        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [0, -1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [1, -1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [2, -1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [3, -1, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [4, -1, 0], [0,0,0,1])
        
        # =====
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [5, -1, 0], [0,0,0.707,0.707])
         
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [5, 0, 0], [0,0,0.707,0.707])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [5, 1, 0], [0,0,0.707,0.707])
         
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [5, 2, 0], [0,0,0.707,0.707])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [5, 3, 0], [0,0,0.707,0.707])
        
        # =====
        
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-4, 4, 0], [0,0,0,1])
 
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-3, 4, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-2, 4, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-1, 4, 0], [0,0,0,1])

        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [0, 4, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [1, 4, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [2, 4, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [3, 4, 0], [0,0,0,1])
        
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [4, 4, 0], [0,0,0,1])
        
        # =====
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-4, -1, 0], [0,0,0.707,0.707])
         
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-4, 0, 0], [0,0,0.707,0.707])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-4, 1, 0], [0,0,0.707,0.707])
         
        wall_id = p.loadURDF('outside_wall.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-4, 2, 0], [0,0,0.707,0.707])
        
        wall_id = p.loadURDF('outside_wall_dark.urdf')
        p.resetBasePositionAndOrientation(
            wall_id, [-4, 3, 0], [0,0,0.707,0.707])
    
    def add_robot(self):
        self.racecar_id = p.loadURDF('racecar.urdf', [0,0,0], [0,0,0,1])
    
    def move_car(self):
        pos = [self.x[0], self.x[1], 0.]
        quat = p.getQuaternionFromEuler([0.,0.,self.x[2]+np.pi])
        p.resetBasePositionAndOrientation(self.racecar_id, pos, quat)
    
    def get_closest_waypoint(self):
        wp = self.waypoints[:,:2]
        pos = np.array([[self.x[0], self.x[1]]])
        d = np.sum((wp - pos)**2, axis=1)
        closest_waypoint = np.argmin(d)
        
        return closest_waypoint, d[closest_waypoint]**0.5
    
    def update_waypoint_drawing(self):
        colors = np.zeros((len(self.waypoints), 3))
        colors[self.current_waypoint] = [1,0,0]
        next_waypoint = (self.current_waypoint + 1)%len(self.waypoints)
        colors[next_waypoint] = [0,1,0]
        
        self.waypoint_id = p.addUserDebugPoints(
            self.waypoints, colors, pointSize=10,
            replaceItemUniqueId=self.waypoint_id,
        )
    
    def render_panorama(self):
        car_pos, car_orient = p.getBasePositionAndOrientation(
            self.racecar_id)
        steering = p.getEulerFromQuaternion(car_orient)[2] + np.pi

        camera_height = 0.2

        # left camera
        left_cam = np.array(car_pos) + [0,0,camera_height]
        left_cam_to = np.array([
            car_pos[0] + np.cos(steering + 1 * np.pi / 2) * 10,
            car_pos[1] + np.sin(steering + 1 * np.pi / 2) * 10,
            car_pos[2] + camera_height,
        ])

        # front camera
        front_cam = np.array(car_pos) + [0,0,camera_height]
        front_cam_to = np.array([
            car_pos[0] + np.cos(steering + 0 * np.pi / 2) * 10,
            car_pos[1] + np.sin(steering + 0 * np.pi / 2) * 10,
            car_pos[2] + camera_height,
        ])

        # right camera
        right_cam = np.array(car_pos) + [0,0,camera_height]
        right_cam_to = np.array([
            car_pos[0] + np.cos(steering + 3 * np.pi / 2) * 10,
            car_pos[1] + np.sin(steering + 3 * np.pi / 2) * 10,
            car_pos[2] + camera_height,
        ])

        # back camera
        back_cam = np.array(car_pos) + [0,0,camera_height]
        back_cam_to = np.array([
            car_pos[0] + np.cos(steering + 2 * np.pi / 2) * 10,
            car_pos[1] + np.sin(steering + 2 * np.pi / 2) * 10,
            car_pos[2] + camera_height,
        ])

        cam_eyes = [left_cam, front_cam, right_cam, back_cam]
        cam_targets = [left_cam_to, front_cam_to, right_cam_to, back_cam_to]
        
        images = []
        for i in range(4):
            # Define the camera view matrix
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_eyes[i],
                cameraTargetPosition=cam_targets[i],
                cameraUpVector = [0,0,1]
            )
            # Define the camera projection matrix
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=90,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            # Add the camera to the scene
            _,_,rgb,depth,segm = p.getCameraImage(
                width = self.resolution,
                height = self.resolution,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_TINY_RENDERER,
            )
            
            images.append(rgb[:,:,:3])

        l,f,r,b = images
        rgb_strip = np.concatenate([l,f,r,b], axis=1)
        rgb_strip = np.concatenate(
            [rgb_strip[:,-self.resolution//2:],
             rgb_strip[:,:-self.resolution//2]],
            axis=1,
        )

        return rgb_strip
    
    def minimized_angle(self, angle):
        """Normalize an angle to [-pi, pi]."""
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle >= np.pi:
            angle -= 2 * np.pi
        return angle

def register_racetrack():
    gym.register('Example-Racetrack-v0', Racetrack)
