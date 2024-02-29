import gymnasium as gym
import numpy as np
import torch

from collections import deque
from torchvision import transforms as T

class PrevStep(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.queue = deque(maxlen=2) 
    
    def prev_step(self):
        return self.queue[0]

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        # enqueue twice if the len is zero
        if (len(self.queue) == 0):
            self.queue.append((obs, reward, term, trunc, info))

        self.queue.append((obs, reward, term, trunc, info))
            
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        self.queue = deque(maxlen=2) 
        return self.env.reset(**kwargs)

class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0

        for _ in range(self._skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info

class Transpose(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        return self.__transpose(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        return self.__transpose(observation), info

    def __transpose(self, observation):
        observation = np.transpose(observation, (2, 0, 1))

        return torch.tensor(observation, dtype=torch.float)

class GrayScale(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)


    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.__to_grayscale(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        return self.__to_grayscale(observation), info

    def __to_grayscale(self, observation):
        grayscale_transform = T.Grayscale()

        return self.__normalize(grayscale_transform(observation))

    def __normalize(self, observation):
        min_val = observation.min()
        max_val = observation.max()

        # Prevent division by zero if max and min are the same
        if min_val == max_val:
            return observation - min_val
        else:
            return (observation - min_val) / (max_val - min_val)

class Rezise(gym.Wrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self._shape = shape 

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        return self.__resize(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        return self.__resize(observation), reward, terminated, truncated, info

    def __resize(self, observation):
        resize_transform = T.Resize(self._shape, antialias=True)

        return resize_transform(observation)

class StackFrames(gym.Wrapper):
    def __init__(self, env, frames):
        super().__init__(env)
        self._frames = frames 

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._stack = torch.cat((self._stack[1:], observation), dim=0) 

        return self._stack, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._stack = observation.repeat(self._frames, 1, 1)

        return self._stack, info


class SuperMarioBrosInfo(gym.Wrapper):
    class _Info:
        def __init__(self, info):
            self._info = info

        def world(self):
            return self._info["levelHi"] + 1

        def level(self):
            return self._info["levelLo"] + 1

        def lives(self):
            return self._info["lives"]

        def xscroll(self):
            return self._info["xscrollHi"] * 256 + self._info["xscrollLo"]

        def score(self):
            return self._info["score"]

        def time(self):
            return self._info["time"]

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        return observation, reward, terminated, truncated, self._Info(info) 

class PunchOutInfo(gym.Wrapper):
    class _Info:
        def __init__(self, info):
            self._info = info 

        def score(self):
            return \
                self._info["score_100000"] * 100_000 + \
                self._info["score_010000"] * 10_000 + \
                self._info["score_001000"] * 1_000 + \
                self._info["score_000100"] * 100 + \
                self._info["score_000010"] * 10 + \
                self._info["score_000001"]
        
        def is_fighting(self):
            return self._info["fight_status"] == 1 

        def is_opp_ko(self):
            return self._info["opponent_ko"] == 1 

        def hearts(self):
            return self._info["hearts_10"] * 10 + self._info["hearts_01"] 

        def stars(self):
            return self._info["stars"] 

        def health(self):
            return self._info["little_mac_health"] 

        def opp_health(self):
            return self._info["opponents_health"] 

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        return observation, reward, terminated, truncated, self._Info(info) 

