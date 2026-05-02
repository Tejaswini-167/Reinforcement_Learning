import gymnasium as gym
import numpy as np


def value_iteration(env):
    num_itns = 1000
    threshold = 1e-20
    gamma = 1.0

    value_table = np.zeros(env.observation_space.n)

    for i in range(num_itns):
        updated_val_tab = np.copy(value_table)

        for s in range(env.observation_space.n):
            q_values = []

            for a in range(env.action_space.n):
                q_value = sum(
                    prob * (reward + gamma * updated_val_tab[next_state])
                    for prob, next_state, reward, done in env.P[s][a]
                )
                q_values.append(q_value)

            value_table[s] = max(q_values)

        if np.sum(np.abs(updated_val_tab - value_table)) <= threshold:
            break

    return value_table


def extract_policy(env, value_table):
    gamma = 1.0
    policy = np.zeros(env.observation_space.n, dtype=int)

    for s in range(env.observation_space.n):
        q_values = []

        for a in range(env.action_space.n):
            q_value = sum(
                prob * (reward + gamma * value_table[next_state])
                for prob, next_state, reward, done in env.P[s][a]
            )
            q_values.append(q_value)

        policy[s] = np.argmax(q_values)

    return policy


# main program
env = gym.make("FrozenLake-v1", render_mode="human")
env.reset()
env.render()

env = env.unwrapped

optimal_value_function = value_iteration(env)
optimal_policy = extract_policy(env, optimal_value_function)

print("Optimal Value Function:")
print(optimal_value_function)

print("\nOptimal Policy:")
print(optimal_policy)