import gymnasium as gym
import pandas as pd

env = gym.make("FrozenLake-v1",render_mode="human")

print(env.action_space)
print(env.observation_space)

env.reset()
env.render()

def random_policy():
    return env.action_space.sample()

V ={}

num_episodes =10
num_steps =5
alpha = 0.1
 
gamma =0.9

for s in range(env.observation_space.n):
    V[s] = 0

for i in range(num_episodes):
    s= env.reset()
    s = s[0]
    for j in range(num_steps):
        a = random_policy()
        s_next,reward,done,truncated,info = env.step(a)

        V[s] += alpha * (reward + gamma * V[s_next] - V[s])

        s=s_next

        if done:
            break
    
    df = pd.DataFrame(list(V.items()),columns=['state','values'])
    print(df)