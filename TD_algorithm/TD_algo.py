import numpy as np
from collections import namedtuple


R = np.array([+10, 0, 0, 0, 0, 0, +1])
state_space = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
action_space = ["Left", "Right"]
Value = np.zeros(7)  
MDP = namedtuple("MDP", ["state", "action", "reward","next_state"])

def model(state_index, action):
    if (action == "Left"):
        return max(0, state_index - 1)
    elif (action == "Right"):
        return min(6, state_index + 1)

def rewardFunc(state_idx):
    reward = R[state_idx]
    return reward

GAMMA = 0.99
ALPHA = 0.01
EPISODE_LENGTH = 100
STEP_LENGTH = 100


class Rover:
    def __init__(self):
        pass
    
    def main(self):
        for _ in range(EPISODE_LENGTH):
            mdp_list = []
            state = 3
            for a in range(STEP_LENGTH):
                self.policy = np.random.randint(0, 2, size=7)
                reward = rewardFunc(state)
                action_idx = self.policy[state] # 0 veya 1
                action = action_space[action_idx]            
                next_state = model(state, action)
                mdp = MDP(state=state, action=action_idx, reward=reward, next_state=next_state)
                mdp_list.append(mdp)
                state = next_state
                if ((state == 0) or (state == 6)):
                    reward = rewardFunc(state)
                    mdp = MDP(state=state, action=action_idx, reward=reward, next_state=next_state)
                    mdp_list.append(mdp)
                    break
            print(mdp_list)
            for i in range(len(mdp_list)):
                s = mdp_list[i].state
                r = mdp_list[i].reward
                next_s = mdp_list[i].next_state
                Value[s] += ALPHA * (r + GAMMA * Value[next_s] - Value[s])
        
        print(Value)

            


if __name__ == "__main__":
    rover = Rover()
    rover.main()
            
                





