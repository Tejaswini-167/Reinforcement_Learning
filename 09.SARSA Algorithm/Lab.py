import gymnasium as gym
import random
import pandas as pd
#create the envt
env = gym.make("FrozenLake-v1", render_mode = "human")
env.reset()
env.render()

#define a dictionary for the Q table. Initialize the Q value of all (s,a) pairs to 0.0
Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0

df1 = pd.DataFrame(list(Q.items()), columns = ['state=action', 'value'])
print(df1)



def epsilon_greedy(state,epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x : Q[(state,x)])


alpha = 0.85
gamma = 0.90
epsilon = 0.8

num_eps = 10
num_steps = 50
#compute the policy for each episode
for i in range(num_eps):
    s = env.reset()
    s = s[0]
    a = epsilon_greedy(s,epsilon)
    for t in range(num_steps):
        s_, r, done, _, _ = env.step(a)
        a_ = epsilon_greedy(s_, epsilon)
        predict = Q[(s,a)]
        target = r + gamma * Q[(s_, a_)]

        Q[(s,a)] = Q[(s,a)] + alpha * (target - predict)

        s = s_
        a = a_
        if done:
            break
df = pd.DataFrame(list(Q.items()), columns = ['state=action', 'value'])
print(df)