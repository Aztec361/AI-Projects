import gym
import random
import numpy as np
import time
from collections import deque
import pickle
from collections import defaultdict

EPISODES = 20000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1
EPSILON_DECAY = 0.999

def default_Q_value():
    return 0

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)

    Q_table = defaultdict(default_Q_value)  # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        state = env.reset()

        while not done:
            if np.random.random() > EPSILON:
                action = np.argmax([Q_table[(state, a)] for a in range(env.action_space.n)])
            else:
                action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)

            # Q-Learning Update Rule
            current_q = Q_table[(state, action)]
            if not done:
                poss_actions = np.max([Q_table[(new_state, a)] for a in range(env.action_space.n)])
                max_future_q = reward + DISCOUNT_FACTOR * poss_actions
                Q_table[(state, action)] = current_q + LEARNING_RATE * (max_future_q - current_q)
            else:
                Q_table[(state, action)] = current_q + LEARNING_RATE * (reward - current_q)

            episode_reward += reward
            state = new_state

        EPSILON *= EPSILON_DECAY  
        episode_reward_record.append(episode_reward)

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record)) / 100))
            print("EPSILON: " + str(EPSILON))

    # Save the trained Q-table and the final value of epsilon
    with open('Q_TABLE.pkl', 'wb') as f:
        pickle.dump([Q_table, EPSILON], f)
