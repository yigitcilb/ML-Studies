import numpy as np
import time

GAMMA = 0.99
ITERATION = 1000

env = np.array([
    [-0.2, -0.2, -0.2, 1],
    [-0.2, np.nan, -0.2, -1],
    [-0.2, -0.2, -0.2, -0.2]
])

class Robot:
    def __init__(self):
        self.actions = ['West','North','East','South']
        self.s0 = [2, 0]
        self.V = np.zeros_like(env)
        self.terminal_states = [(0,3), (1,3)]

        self.rows, self.cols = env.shape

    def Prob(self, action): 
        if action == 'West': return np.array([0.8, 0.1, 0, 0.1]) 
        elif action == 'North': return np.array([0.1, 0.8, 0.1, 0]) 
        elif action == 'East': return np.array([0, 0.1, 0.8, 0.1]) 
        elif action == 'South': return np.array([0.1, 0, 0.1, 0.8])

    def next_state(self, i, j, action):
        ni, nj = i, j
        if (i,j) in self.terminal_states:
            return (i,j)
        if action == 'West' and j > 0 and not np.isnan(env[i,j-1]):
            nj = j-1
        elif action == 'East' and j < self.cols-1 and not np.isnan(env[i,j+1]):
            nj = j+1
        elif action == 'North' and i > 0 and not np.isnan(env[i-1,j]):
            ni = i-1
        elif action == 'South' and i < self.rows-1 and not np.isnan(env[i+1,j]):
            ni = i+1
        return (ni, nj) 

    def main(self):
        print("Initial V at s0:", self.V[self.s0[0], self.s0[1]])

        for _ in range(ITERATION):
            V_new = np.copy(self.V)
            for i in range(self.rows):
                for j in range(self.cols):
                    if np.isnan(env[i,j]):
                        continue
                    if (i,j) in self.terminal_states:
                        V_new[i,j] = env[i,j]
                        continue
                    
                    best_idx = 0
                    value = 0
                    for idx, act in enumerate(self.actions): 
                        probs = self.Prob(act)
                        value = 0
                        deger = -9999999
                        for prob_idx, next_act in enumerate(self.actions):
                            ni, nj = self.next_state(i, j, next_act)
                            value += probs[prob_idx] * self.V[ni, nj]
                        if (value > deger):
                            best_idx = idx
                            deger = value

                    V_new[i,j] = env[i,j] + GAMMA * deger


            self.V = V_new

        print("Value Function after Value Iteration:")
        print(self.V)


if __name__ == "__main__":
    robot = Robot()
    robot.main()
