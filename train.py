# train the PPO agent for the cartpole task family

from env import cartpole
from stable_baselines3 import PPO

env = cartpole.CartPoleEnv()
# env = gym.make('CartPole-v1')
model = PPO("MlpPolicy", env, verbose=0)


for _ in range(200):
    # learn
    model.learn(total_timesteps=int(1e4))
    # eval
    for _ in range(5):
        obs = env.reset()
        episode_length = 0
        while True:
            episode_length += 1
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
        env.close()
        print('keep up time', episode_length)

model.save("trained_agent/ppo_baseline_0330") 
# TODO, save the config with the model into one file.
