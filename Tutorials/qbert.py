import gymnasium
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque 
import random 

def build_q_network(input_shape, num_actions):
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model 

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add_experience(self, experience):
        self.buffer.append(experience)
    
    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

class QAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size 
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 32
        self.memory = ReplayBuffer(buffer_size=1000)
        self.model = build_q_network(input_shape=(state_size), num_actions=action_size)
    
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.model.predict(state.reshape(1, -1))
            return np.argmax(q_values[0])
    
    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return 
        
        batch = self.memory.sample_batch(self.batch_size)

        for state, action, reward, next_state, done in batch:
            target = reward 

            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0]))
            
            q_values = self.model.predict(state.reshape(1, -1))

            q_values[0][action] = target 

            self.model.fit(state.reshape(1, -1), q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

env = gymnasium.make('Qbert-v0')
state_size = np.prod(np.array(env.observation_space.shape))
action_size = env.action_space.n 
agent = QAgent(state_size, action_size)

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    state = state.flatten()

    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.flatten()

        agent.memory.add_experience((state, action, reward, next_state, done))
        agent.train()

        total_reward += reward
        state = next_state 

        if done:
            break 
    
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")