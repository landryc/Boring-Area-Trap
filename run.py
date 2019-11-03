

import numpy as np
import matplotlib.pyplot as plt
from broken_armed_bandit import BrokenArmedBandit
from q_learning import QLearning
from bins_asrn import BinsASRN
import time

# seed = int(time.time())
seed = 2
np.random.seed(seed)



def good_choice(all_goods):
	all_goods = all_goods.ravel()
	return (all_goods.sum() / len(all_goods)) * 100

def printQtable(n_figure, all_q_tables, index_player, title, name):
	plt.figure(n_figure)
	q_table = all_q_tables[index_player]
	q_table = np.asarray(q_table)[:, 1:]
	plt.plot(q_table[:, 0], '-g')
	plt.plot(q_table[:, 1], '-r')
	plt.legend(['Q r', 'Q l'])
	plt.title(f'{title} {index_player}')
	plt.xlabel('Steps')
	plt.ylabel('Q-values')
	plt.savefig(f'{name}')


def demonstrate_boring_areas_trap(game_length=500):
    n_players = 3
    left_arm_mean = 0.
    right_arm_mean = 1.
    left_arm_std = 0.5
    right_arm_std = 7.
    
    file = 'Images_boring_areas_trap_' + str(game_length)

    import os
    cmd = 'ls' if os.name == 'posix' else 'dir'
    if os.system(f'{cmd} {file}'):
    	os.system(f'mkdir {file}')

    seed = int(time.time())
    # index_player = np.random.randint(0, n_players)
    index_player = 2

    # Pure Problem
    use_asrn = False
    epsilon = 0.0
    all_q_tables, all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = 0.1, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999)
    print(good_choice(all_goods))
    printQtable(n_figure=1, all_q_tables=all_q_tables, index_player=index_player, title='Q-table without ASRN of player n°', name=f'{file}' + '/Qtable_without_asrn.png')

    # Exploration
    use_asrn = False
    epsilon = 1.0
    all_q_tables, all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = 0.1, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999)
    print(good_choice(all_goods))
    printQtable(n_figure=2, all_q_tables=all_q_tables, index_player=index_player, title='Q-table without ASRN and with Exploration of player n°', name=f'{file}' + '/Qtable_with_only_Exploration_asrn.png')

    # ASRN
    use_asrn = True
    epsilon = 1.0
    all_q_tables, all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = 0.1, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999)
    print(good_choice(all_goods))
    printQtable(n_figure=3, all_q_tables=all_q_tables, index_player=index_player, title='Q-table with ASRN of player n°', name=f'{file}' + '/Qtable_with_asrn.png')

    plt.show()


def demonstrate_low_alpha(game_length=1000000):
    n_players = 1

    left_arm_mean = 0.
    right_arm_mean = 0.001
    left_arm_std = 0.1
    right_arm_std = 13.
    learning_rate = 0.000001 # low alpha

    file = 'Images_low_alpha_' + str(learning_rate)
    
    import os
    cmd = 'ls' if os.name == 'posix' else 'dir'
    if os.system(f'{cmd} {file}'):
    	os.system(f'mkdir {file}')


    # Détermination de l'index_player des Q-tables à afficher.
    index_player = np.random.randint(0, n_players)
    # index_player = 2

    seed = 2

    # Pure problem
    np.random.seed(seed)
    use_asrn = False
    epsilon = 0.0 # No exploration to see the pure problem.
    all_q_tables, all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = learning_rate, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999)
    printQtable(n_figure=1, all_q_tables=all_q_tables, index_player=index_player, title='Q-table without ASRN of player n°', name=f'{file}' + '/Qtable_without_asrn.png')

    # Exploration
    np.random.seed(seed)
    use_asrn = False
    epsilon = 1. 
    all_q_tables, all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = learning_rate, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999)    
    printQtable(n_figure=2, all_q_tables=all_q_tables, index_player=index_player, title='Q-table without ASRN and with Exploration of player n°', name=f'{file}' + '/Qtable_with_only_Exploration_asrn.png')

    # ASRN
    np.random.seed(seed)
    use_asrn = True
    epsilon = 1.0 
    all_q_tables, all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = learning_rate, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999)
    printQtable(n_figure=3, all_q_tables=all_q_tables, index_player=index_player, title='Q-table with ASRN of player n°', name=f'{file}' + '/Qtable_with_asrn.png')

    plt.show()
 

def run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn, learning_rate = 0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.99):
    all_rewards = []
    all_goods = []
    all_losses = []
    all_q_tables = []
    trained_agent_q_values = [left_arm_mean / (1 - gamma), right_arm_mean / (1 - gamma)]
 
    for j in range(n_players):
        two_armed_bandit = BrokenArmedBandit(left_arm_mean=left_arm_mean, right_arm_mean=right_arm_mean, left_arm_std=left_arm_std, right_arm_std=right_arm_std)

        ## giving the real mean as initialization(!)
        left_initial_mean = trained_agent_q_values[0]
        right_initial_mean = trained_agent_q_values[1]

        q_learning = QLearning(left_initial_mean, right_initial_mean, learning_rate, gamma, epsilon, epsilon_decay)

        rewards = np.zeros((game_length, 1))
        goods = np.zeros((game_length, 1))
        losses = np.zeros((game_length, 1))
        q_table = []

        if use_asrn:
            asrn = BinsASRN(0, learning_period=game_length/10)
        for i in range(game_length):
            right, reward_estimation = q_learning.choose()
            good = q_learning.right_mean > q_learning.left_mean
            goods[i] = good

            q_table.append([right, q_learning.right_mean, q_learning.left_mean])
            
            reward = two_armed_bandit.pull(right)
            rewards[i] = reward

            if use_asrn:
                if right:
                    updated_right_mean = (1 - q_learning.learning_rate) * q_learning.right_mean + q_learning.learning_rate * (reward + q_learning.gamma * q_learning.right_mean)
                    reward = asrn.noise(q_learning.right_mean, updated_right_mean, reward)
                else:
                    updated_left_mean = (1 - q_learning.learning_rate) * q_learning.left_mean + q_learning.learning_rate * (reward + q_learning.gamma * q_learning.left_mean)
                    reward = asrn.noise(q_learning.left_mean, updated_left_mean, reward)

            loss = q_learning.update(right, reward)
            losses[i] = loss

        all_rewards.append(rewards)
        all_goods.append(goods)
        all_losses.append(losses)
        all_q_tables.append(q_table)

    return all_q_tables, np.asarray(all_rewards), np.asarray(all_goods), np.asarray(all_losses)

if __name__ == '__main__':
    demonstrate_boring_areas_trap()
    demonstrate_low_alpha()