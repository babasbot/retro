#!/usr/bin/env python

from agent import Agent, action_space
from wrappers import Transpose, SkipFrames, Rezise, GrayScale, StackFrames, PrevStep, SuperMarioBrosInfo 
from rewards import calculate_reward
from logger import MetricLogger
from pathlib import Path

import retro
import datetime

env_config = {
    "game": "SuperMarioBros-Nes",
    # "render_mode": "rgb_array",
    "render_mode": "human",
    "record": False,
}

def make_env():
    env = retro.make(**env_config)
    env = SkipFrames(env, skip=4)
    env = Transpose(env) 
    env = Rezise(env, shape=(84,84))
    env = GrayScale(env)
    env = StackFrames(env, frames=4)
    env = SuperMarioBrosInfo(env)
    env = PrevStep(env)

    return env

def main():
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    env = make_env()
    agent = Agent(state_space=(4,84,84), save_dir=save_dir)

    logger = MetricLogger(save_dir)
    episodes = 40_000

    for episode in range(episodes):
        state, _ = env.reset()

        while True:
            action_idx = agent.act(state)
            action = action_space[action_idx]

            next_state, _, terminated, truncated, next_info = env.step(action)
            state, _, _, _, info = env.prev_step()

            reward = calculate_reward(info=info, next_info=next_info)
            
            agent.remember(state, action_idx, reward, next_state, terminated or truncated)

            loss, q = agent.learn()
            logger.log_step(reward, loss, q)

            if terminated or truncated:
                break

        logger.log_episode()
        logger.record(episode=episode, epsilon=agent.epsilon, step=agent.current_step)

    env.close()

if __name__ == "__main__":
    main()
