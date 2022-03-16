import numpy as np
import random
from matplotlib.pyplot import * 


class Agent():
    """
    An interface of agent object

    ...

    Attributes
    ----------
    setting : Dict()
        Dict of all agent setting
    state : int
        Position of the agent on the grid
    last_action : int
        Last movement done by the agent

    Methods
    -------
    next_step():
        Get next move of the agent
    update_table(is_training=True):
        Update q-table of the agent
    display_table():
        Show matrix plot
    """
    def __init__(self, setting):
        self.setting     = setting
        self.state       = 0
        self.last_action = 0 


    def next_step(self):
        raise NotImplementedError


    def update_table(self, new_state, reward, is_training):
        raise NotImplementedError

    
    def display_table(self):
        raise NotImplementedError


class AgentSimpleQLearn(Agent):
    def __init__(self, setting):
        super().__init__(setting)
        self.q_table = np.zeros((setting["state_count"],4))
        

    def next_step(self):
        if random.uniform(0, 1) < self.setting["epsilon"]:
            action = np.random.randint(0,4)
        else:
            action = np.argmax(self.q_table[self.state, :])
        self.last_action = action
        self.setting["epsilon"] = max(self.setting["epsilon_min"], self.setting["epsilon"]*self.setting["epsilon_decay"])
        return action


    def update_table(self, new_state, reward, is_training):
        state       = self.state
        action      = self.last_action
        alpha       = self.setting["alpha"]
        gamma       = self.setting["gamma"]
        old_q_value = self.q_table[state, action]
        max_q_value = np.max(self.q_table[new_state, :])

        self.state = new_state
        if is_training:
            self.q_table[state, action] = old_q_value + alpha * (reward + gamma * max_q_value - old_q_value)


    def display_table(self):
        figure()
        imshow(self.q_table,cmap=get_cmap('RdYlGn'))
        title("Q-Tabel")
        draw()

class AgentDoubleQLearn(Agent):
    def __init__(self, setting):
        super().__init__(setting)
        self.q_table       = [np.zeros((setting["state_count"],4)),np.zeros((setting["state_count"],4))]
        self.q_table_index = 0


    def next_step(self):
        if random.uniform(0, 1) < self.setting["epsilon"]:
            action = np.random.randint(0,4)
        else:
            action = np.argmax(self.q_table[self.q_table_index][self.state, :])
        self.last_action = action
        self.setting["epsilon"] = max(self.setting["epsilon_min"], self.setting["epsilon"]*self.setting["epsilon_decay"])
        return action


    def update_table(self, new_state, reward, is_training):
        state         = self.state
        action        = self.last_action
        alpha         = self.setting["alpha"]
        gamma         = self.setting["gamma"]
        old_q_value   = self.q_table[self.q_table_index][state, action]
        best_action_a = np.argmax(self.q_table[self.q_table_index][new_state, :])
        q_value_b     = np.max(self.q_table[(self.q_table_index+1)%2][new_state, best_action_a])

        self.state = new_state
        if is_training:
            self.q_table[self.q_table_index][state, action] = old_q_value + alpha * (reward + gamma * q_value_b - old_q_value)
        self.q_table_index = (self.q_table_index+1)%2
        

    def display_table(self):
        figure()
        subplot(1, 2, 1)
        imshow(self.q_table[0],cmap=get_cmap('RdYlGn'))
        title("Q-Tabel A")
        subplot(1, 2, 2)
        imshow(self.q_table[1],cmap=get_cmap('RdYlGn'))
        title("Q-Tabel B")
        draw()
