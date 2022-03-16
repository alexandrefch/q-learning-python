import gym
from agent import *
from progressBar import *
from matplotlib.pyplot import * 

class Frozenlake():
    """
    A class to manage a frozen lake from gym lib and make an agent play.

    ...

    Attributes
    ----------
    setting : Dict()
        Dict of all game setting
    agent : Agent()
        Agent object that will play and learn
    env : Env()
        Object that store all the gym environment

    Methods
    -------
    reset():
        Reset gym environment 
    play(is_training=True):
        Make agent play until end of the game
    train():
        Make agent play for specific amount of round
    test():
        Make agent play during 1000 game to make statistics
    """


    def __init__(self, game_setting, ai_setting):
        """
        Create agent object using ai_setting and set gym environment up

        Parameters
        ----------
            game_setting : Dict()
                Dictionnary of all game setting
            ai_setting : Dict()
                Dictionnary of all agent setting
        """
        game_setting["lake_type"] = "FrozenLake-v1" if game_setting["lake_size"] == "4x4" else "FrozenLake8x8-v1"
        self.cell_count = 16 if game_setting["lake_size"] == "4x4" else 64
        ai_setting["state_count"] = self.cell_count

        if(ai_setting["q_learning_type"] == "simple"):
            self.agent = AgentSimpleQLearn(ai_setting)
        else:
            self.agent = AgentDoubleQLearn(ai_setting)

        self.setting = game_setting
        self.setting["lake_type"] = "FrozenLake-v1" if game_setting["lake_size"] == "4x4" else "FrozenLake8x8-v1"
        self.env = gym.make(self.setting["lake_type"])
        self.reset()


    def reset(self):
        """
        Reset gym environment

        """
        self.env.reset()
    

    def play(self, is_training=True):
        """
        Make agent play until end of the game

        Parameters
        ----------
            is_training : boolean
                true if the agent will learn and update his q-table
        """
        while True:
            action = self.agent.next_step()
            new_state, _, is_done, _ = self.env.step(action)

            reward = 0
            is_win = new_state == self.cell_count - 1

            if is_done:
                reward = self.setting["reward"] if is_win else self.setting["punish"]

            self.agent.update_table(new_state,reward,is_training)

            if is_done:
                return new_state == self.cell_count - 1
            

    def train(self):
        """
        Make agent play for specific amount of round

        """
        wins_rate = [0]
        rounds = [0]
        win_amount = 0
        for count in progressbar(range(self.setting["round_to_train"]),"Entrainement",33):
            self.reset()
            is_win = self.play()

            if is_win:
                win_amount += 1

            if (count+1) % int(self.setting["round_to_train"] / 100) == 0:
                wins_rate.append(win_amount*100/count)
                rounds.append(count)

        figure()
        plot(rounds,wins_rate)
        title("Win rate over training time")
        ylim(0,100)
        xlim(0,self.setting["round_to_train"])
        draw()


    def test(self):
        """
        Make agent play during 1000 game to make statistics

        """
        win_amount = 0
        for _ in progressbar(range(1000),"Test",33):
            self.reset()
            is_win = self.play(False)
            
            if is_win:
                win_amount += 1
            
        return win_amount * 100 / 1000 
                