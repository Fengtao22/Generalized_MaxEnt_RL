#### Check if torch uses GPU

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

'''
#### Check gym envs
import gym
env = gym.make('Humanoid-v2')

from gym import envs
print(envs.registry.all())    # print the available environments

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

for i_episode in range(200):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()    # take a random action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
'''

### Check the env is determinstic or stochastic 
## https://github.com/openai/gym/issues/1193
import numpy as np
import pytest

import gym
env_names = ["HalfCheetah-v2", "Ant-v2", "Hopper-v2", "Reacher-v2", "Walker2d-v2", "Swimmer-v2"]

for env_name in env_names:
    for _ITER in range(10):
        print(_ITER)
        states = []
        actions = []

        env = gym.make(env_name)
        env.seed(1234)
        o0 = env.reset()
        # env.render()
        for i in range(100):
            s = env.env.sim.get_state().flatten()
            states.append(s)
            a = env.action_space.sample()
            actions.append(a)
            env.step(a)
            # env.render()
        print('len(states)', len(states))
        s = env.env.sim.get_state().flatten()
        states.append(s)

        print("\nPLAYBACK from s0 \n")

        env = gym.make(env_name)
        env.seed(1234)
        env.reset()
        env.env.sim.set_state_from_flattened(states[0])
        env.env.sim.forward()
        # env.render()

        for i in range(100):
            env.step(actions[i])
            # env.render()

        print("state s is {}".format(env.env.sim.get_state().flatten()))
        print("state s should be {}".format(states[-1]))
        assert(np.allclose(env.env.sim.get_state().flatten(), states[-1]))

        print("\nPLAYBACK from s1 \n")

        env = gym.make(env_name)
        env.seed(1234)
        env.reset()
        env.env.sim.set_state_from_flattened(states[1])
        env.env.sim.forward()
        # env.render()

        for i in range(99):
            env.step(actions[i + 1])
            # env.render()


        print("state s is {}".format(env.env.sim.get_state().flatten()))
        print("state s should be {}".format(states[-1]))
        assert(np.allclose(env.env.sim.get_state().flatten(), states[-1]))


        print("\nPLAYBACK from s0, but force state at s1 \n")

        env = gym.make(env_name)
        env.seed(1234)
        env.reset()
        env.env.sim.set_state_from_flattened(states[0])
        env.step(actions[0])
        assert(np.allclose(env.env.sim.get_state().flatten(), states[1]))
        env.env.sim.set_state_from_flattened(states[1])
        env.env.sim.forward()
        # env.render()

        for i in range(99):
            env.step(actions[i + 1])
            # env.render()


        print("state s is {}".format(env.env.sim.get_state().flatten()))
        print("state s should be {}".format(states[-1]))
        assert(np.allclose(env.env.sim.get_state().flatten(), states[-1]))

        print("All tests passed!\n\n\n")

