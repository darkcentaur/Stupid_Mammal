import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env = gym.make("Pendulum-v1", g=9.81, render_mode='rgb_array')
env = RecordVideo(env, "video", episode_trigger=lambda episode_id: True)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
