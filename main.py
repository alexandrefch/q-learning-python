import argparse
from frozenlake import Frozenlake
from matplotlib.pyplot import * 

ASCII = """   ____                          __         __       
  / __/____ ___  ___ ___  ___   / /  ___ _ / /__ ___ 
 / _/ / __// _ \/_ // -_)/ _ \ / /__/ _ `//  '_// -_)
/_/  /_/   \___//__/\__//_//_//____/\_,_//_/\_\ \__/ 
"""

LINE = '═'*55+'\n'

if __name__ == '__main__':

    print(LINE+ASCII+'\n'+LINE)

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", nargs='?', help="Learning rate", type=float, default=0.1)
    parser.add_argument("--q_type", nargs='?', help="Q-Learning type 'simple' or 'double'", type=str, default="simple")    
    parser.add_argument("--gamma", nargs='?', help="Discount factor", type=float, default=0.95)    
    parser.add_argument("--epsilon", nargs='?', help="Epsilon start value", type=float, default=1)    
    parser.add_argument("--epsilon_decay", nargs='?', help="Decay ration of epsilon", type=float, default=0.95)    
    parser.add_argument("--epsilon_min", nargs='?', help="Min value of epsilon", type=float, default=0.05)
    parser.add_argument("--lake_size", nargs='?', help="Lake type from lib gym", type=str, default="4x4")
    parser.add_argument("--try_amount", nargs='?', help="Amount of try to train the AI", type=int, default=1000)
    parser.add_argument("--reward", nargs='?', help="Value of the reward", type=float, default=1)
    parser.add_argument("--punish", nargs='?', help="Value of the punishment", type=float, default=-1)
    args = parser.parse_args()

    ai_setting = {
        "q_learning_type" : args.q_type,
        "alpha"           : args.alpha,
        "gamma"           : args.gamma,
        "epsilon"         : args.epsilon,
        "epsilon_decay"   : args.epsilon_decay,
        "epsilon_min"     : args.epsilon_min
    }

    game_setting = {
        "lake_size" : args.lake_size,
        "round_to_train" : args.try_amount,
        "reward" : args.reward,
        "punish" : args.punish
    }

    print("Setting :\n")
    for setting_dict in [ai_setting, game_setting]:
        for key, value in setting_dict.items():
            print("%s %s"%(key.ljust(15),value))
    print('')

    game = Frozenlake(game_setting, ai_setting)

    print(LINE)

    game.train()
    result = game.test()

    print('\n' + LINE)

    print("Win rate %.2f%%" % result)

    print('\n' + LINE)
    
    game.agent.display_table()

    show()