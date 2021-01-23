############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Reinforcement Learning
# Lesson: Q-Learning - Tic Tac Toe

# Citation: 
# PEREIRA, V. (2019). Project: Deep Reinforcement Learning, File: Python-AI-QL-Tic-Tac-Toe.py, GitHub repository:<https://github.com/Valdecy/Reinforcement Learning>

############################################################################

# Importing Required Libraries
import pandas as pd
import numpy  as np
import csv
import math
import os

# Loading Data
states_table  = pd.read_csv('Tic-Tac-Toe-States.txt', sep = '\t')
states_random = pd.read_csv('Tic-Tac-Toe-Random.txt', sep = '\t')
states_table  = states_table.values.tolist()
states_random = states_random.values.tolist()

# Preparing Q-table X and Q-Table O
def set_q_tables(states_table):
    q_table_x = pd.DataFrame(np.random.randn(len(states_table), 9)/1000).astype('float64')
    q_table_o = pd.DataFrame(np.random.randn(len(states_table), 9)/1000).astype('float64') 
    q_table_x = q_table_x.values.tolist()
    q_table_o = q_table_o.values.tolist()
    for i in range(0, len(q_table_x)):
        for j in range(0, len(q_table_x[0])):
            if(states_table[i][-1] == 2):
                q_table_o[i][j] = np.nan
            elif(states_table[i][-1] == 1):
                q_table_x[i][j] = np.nan
            if(states_table[i][j + 2] == states_table[i][0]):
                q_table_x[i][j] = np.nan
                q_table_o[i][j] = np.nan
    return q_table_x, q_table_o

# Update Q-Value
def update_q(q_table_mine, q_table_other, states_table, state = 0, action = 0, reward_col = 11, alpha = 0.2, gamma = 0.9):
    next_state_mine  = states_table[state][action + 2]
    action_nso = 0
    try:
        action_nso  = np.nanargmax(q_table_other[next_state_mine])
    except:
        action_nso  = 0
    next_state_other = states_table[next_state_mine][action_nso + 2]
    
    np.warnings.filterwarnings('ignore')

    if (math.isnan(np.nanmax(q_table_mine[next_state_mine]))):
        max_mine_nsm = 0.0
    else: 
        max_mine_nsm = np.nanmax(q_table_mine[next_state_mine])
        
    if (math.isnan(np.nanmax(q_table_mine[next_state_other]))):
        max_mine_nso = 0.0
    else: 
        max_mine_nso = np.nanmax(q_table_mine[next_state_other])
    
    if (math.isnan(q_table_mine[state][action])):
        return np.nan
        
    if (states_table[next_state_mine][reward_col] == 5 or states_table[next_state_mine][reward_col] == 100):
        q_table_mine[state][action] = (1 - alpha)*q_table_mine[state][action] + alpha*(states_table[next_state_mine][reward_col] + gamma* max_mine_nsm)
    else:
        q_table_mine[state][action] = (1 - alpha)*q_table_mine[state][action] + alpha*(states_table[next_state_other][reward_col] + gamma* max_mine_nso)
    return q_table_mine[state][action]

# Choose a Legal Random Action for a Specific State    
def random_action(states_random, state = 0):
    action = 0
    rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    if (rand < states_random[state][1]):
        action = 0
    elif(rand >= states_random[state][1] and rand < states_random[state][2]):
        action = 1
    elif(rand >= states_random[state][2] and rand < states_random[state][3]):
        action = 2
    elif(rand >= states_random[state][3] and rand < states_random[state][4]):
        action = 3
    elif(rand >= states_random[state][4] and rand < states_random[state][5]):
        action = 4
    elif(rand >= states_random[state][5] and rand < states_random[state][6]):
        action = 5
    elif(rand >= states_random[state][6] and rand < states_random[state][7]):
        action = 6
    elif(rand >= states_random[state][7] and rand < states_random[state][8]):
        action = 7
    elif(rand >= states_random[state][8] and rand < states_random[state][9]):
        action = 8
    return action

# Play X
def play_x(q_table_x, q_table_o, states_random, states_table, state = 0, cutoff = 1.0, alpha = 0.2, gamma = 0.9):
    rdn = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    action = 0
    if (state == 0):
        rdn = 0
    if (rdn < cutoff):
        action = random_action(states_random, state)           
    elif(rdn >= cutoff):
        try:
            action = np.nanargmax(q_table_x[state])
        except:
            action = np.nan
    if (math.isnan(action) or states_table[state][11] == 5 or states_table[state][11] == 100 or states_table[state][11] == -100):
        next_state = 0
        return q_table_x, next_state   
    q_table_x[state][action] = update_q(q_table_x, q_table_o, states_table, state = state, action = action, reward_col = 11, alpha = 0.2, gamma = 0.9) # 
    next_state = states_table[state][action + 2]
    return q_table_x, next_state

# Play O
def play_o(q_table_o, q_table_x, states_random, states_table, state = 0, cutoff = 1.0, alpha = 0.2, gamma = 0.9):
    rdn = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    action = 0
    if (state == 0):
        rdn = 0
    if (rdn < cutoff):
         action = random_action(states_random, state)
    elif(rdn >= cutoff):
        try:
            action = np.nanargmax(q_table_o[state])
        except:
            action = np.nan
    if (math.isnan(action) or states_table[state][12] == 5 or states_table[state][12] == 100 or states_table[state][12] == -100):
        next_state = 0
        return q_table_o, next_state       
    q_table_o[state][action] =  update_q(q_table_o, q_table_x, states_table, state = state, action = action, reward_col = 12, alpha = alpha, gamma = gamma) #
    next_state = states_table[state][action + 2] #
    return q_table_o, next_state   
 
# View Game Play
def game_string(string, view_board = False):
    string = string.replace('T', '')
    string = string.replace('0', '-')
    string = string.replace('1', 'X')
    string = string.replace('2', 'O')
    string = string[:3] + ' . ' + string[3:]
    string = string[:9] + ' . ' + string[9:]
    board = """
     {} | {} | {}
    -----------
     {} | {} | {}
    -----------
     {} | {} | {}
     """.format(string[0], string[1], string[2], string[6], string[7], string[8], string[12], string[13], string[14])
    if (view_board == True):
        print(board)
    return string

# Play AI X
def play_aix(q_table_x, states_random, states_table, next_state = 0, rdn_aix = False):
    action = 0
    if (rdn_aix == True):
        action = random_action(states_random, next_state)
    elif(rdn_aix == False):
        try:
            action = np.nanargmax(q_table_x[next_state])
        except:
            action = np.nan
    if (math.isnan(action) or states_table[next_state][11] == 5 or states_table[next_state][11] == 100 or states_table[next_state][11] == -100):
        next_state = 0
        return next_state    
    next_state = states_table[next_state][action + 2]
    return next_state

# Play AI O
def play_aio(q_table_o, states_random, states_table, next_state = 0, rdn_aio = False):
    action = 0
    if (rdn_aio == True):
        action = random_action(states_random, next_state)
    elif(rdn_aio == False):
        try:
            action = np.nanargmax(q_table_o[next_state])
        except:
            action = np.nan
    if (math.isnan(action) or states_table[next_state][12] == 5 or states_table[next_state][12] == 100 or states_table[next_state][12] == -100):
        next_state = 0
        return next_state   
    next_state = states_table[next_state][action + 2]
    return next_state

# Spy vs Spy
def play_aix_vs_aio(states_table, states_random, q_table_x, q_table_o, rdn_aix = False, rdn_aio = False, verbose = True, number_of_games = 1000):
    next_state  = 0
    count       = 0
    win_x       = 0
    win_o       = 0
    draw        = 0
    x_turn      = True
    o_turn      = False  
    history     = []
    while count < number_of_games: 
        if (x_turn == True and o_turn == False):
            if (next_state == 0):
                next_state = play_aix(q_table_x, states_random, states_table, next_state = next_state, rdn_aix = True)
            else:
                next_state = play_aix(q_table_x, states_random, states_table, next_state = next_state, rdn_aix = rdn_aix)
            history.append(game_string(states_table[next_state][1], view_board = verbose))
            if (states_table[next_state][11] == 100):
                win_x = win_x + 1
                next_state = 0
                history.append(game_string(states_table[next_state][1], view_board = verbose))
            elif (states_table[next_state][11] == -100):
                win_o = win_o + 1
                next_state = 0
                history.append(game_string(states_table[next_state][1], view_board = verbose))
            elif (states_table[next_state][11] == 5):
                draw = draw + 1
                next_state = 0
                history.append(game_string(states_table[next_state][1], view_board = verbose))
            if (next_state == 0):
                count = count + 1
                x_turn = True
                o_turn = False
        if(next_state == 0):
            x_turn = True
            o_turn = False
        else:
            x_turn = False
            o_turn = True
        if (x_turn == False and o_turn == True): 
            next_state = play_aio(q_table_o, states_random, states_table, next_state = next_state , rdn_aio =  rdn_aio)
            history.append(game_string(states_table[next_state][1], view_board = verbose))
            if (states_table[next_state][12] == 100):
                win_o = win_o + 1
                next_state = 0
                history.append(game_string(states_table[next_state][1], view_board = verbose))
            elif (states_table[next_state][12] == -100):
                win_x = win_x + 1
                next_state = 0
                history.append(game_string(states_table[next_state][1], view_board = verbose))
            elif (states_table[next_state][12] == 5):
                draw = draw + 1
                next_state = 0
                history.append(game_string(states_table[next_state][1], view_board = verbose))
            if (next_state == 0):
                count = count + 1
                x_turn = True
                o_turn = False
        x_turn = True
        o_turn = False
    print('          X win: ', win_x, ' O win: ', win_o, ' Draw: ', draw)
    return history

# Check AI Performance
def ai_check(states_table, states_random, q_table_x, q_table_o):
    print('X (AI) vs O (Random): ')
    simulation = play_aix_vs_aio(states_table, states_random, q_table_x, q_table_o, rdn_aix = False, rdn_aio = True,  verbose = False, number_of_games = 1000)
    print('X (Random) vs O (AI): ')
    simulation = play_aix_vs_aio(states_table, states_random, q_table_x, q_table_o, rdn_aix = True,  rdn_aio = False, verbose = False, number_of_games = 1000)
    print('X (AI) vs O (AI): ')
    simulation = play_aix_vs_aio(states_table, states_random, q_table_x, q_table_o, rdn_aix = False, rdn_aio = False, verbose = False, number_of_games = 1000)
    del simulation
    return

# Export to .js File
def js_python(q_table_x, q_table_o):
    if os.path.exists('Tic-Tac-Toe-q-table-x.js'):
        os.remove('Tic-Tac-Toe-q-table-x.js')
    f = open('Tic-Tac-Toe-q-table-x.js', 'w+')
    f.write('var qtx = [  \r\n ')
    for i in range(0, len(q_table_x)):
        f.write('[')
        for j in range(0, len(q_table_x[0])):
            if (i < len(q_table_x) - 1):
                if(math.isnan(q_table_x[i][j])):
                    f.write('NaN' + ', ')
                else:
                    f.write(str(q_table_x[i][j]) + ', ')
                if (j == len(q_table_x[0]) - 1):
                    f.write('],\r')
            else:
                f.write(str(q_table_x[i][j]) + ', ')
                if (j == len(q_table_x[0]) - 1):
                    f.write('] \r\n  ];')
    f.close() 
    if os.path.exists('Tic-Tac-Toe-q-table-o.js'):
        os.remove('Tic-Tac-Toe-q-table-o.js')
    f = open('Tic-Tac-Toe-q-table-o.js', 'w+')
    f.write('var qto = [  \r\n ')
    for i in range(0, len(q_table_o)):
        f.write('[')
        for j in range(0, len(q_table_o[0])):
            if (i < len(q_table_o) - 1):
                if(math.isnan(q_table_o[i][j])):
                    f.write('NaN' + ', ')
                else:
                    f.write(str(q_table_o[i][j]) + ', ')
                if (j == len(q_table_x[0]) - 1):
                    f.write('],\r')
            else:
                f.write(str(q_table_o[i][j]) + ', ')
                if (j == len(q_table_o[0]) - 1):
                    f.write('] \r\n  ];')
    f.close()
    return

# RL: Q-Learning Algorithm
def q_learning(states_table, states_random, episodes = 150000, alpha = 0.2, gamma = 0.9, save = 100, check = 1000): 
    q_table_x, q_table_o = set_q_tables(states_table)
    count       = 0
    cutoff      = 0
    next_state  = 0
    x_turn      = True
    o_turn      = False 
    print('Episode: ', count)
    while count < episodes:
        if (count < 0.25*episodes):
            cutoff = 0.00
        elif (count >= 0.25*episodes and count < 0.50*episodes):
            cutoff = 0.25
        elif (count >= 0.50*episodes and count < 0.75*episodes):
            cutoff = 0.50
        elif (count >= 0.75*episodes):
             cutoff = 0.75            
        if (x_turn == True and o_turn == False):
            q_table_x, next_state = play_x(q_table_x, q_table_o, states_random, states_table, state = next_state, cutoff = cutoff, alpha = alpha, gamma = gamma)
            x_turn = False
            o_turn = True
            if (next_state == 0):
                count = count + 1 
                x_turn = True
                o_turn = False
        if (x_turn == False and o_turn == True): 
            q_table_o, next_state = play_o(q_table_o, q_table_x, states_random, states_table, state = next_state, cutoff = cutoff, alpha = alpha, gamma = gamma)
            x_turn = True
            o_turn = False
            if (next_state == 0):
                count = count + 1  
        if (next_state == 0 and (count % save == 0 or count > episodes)):
            print('Episode:', count)
            with open('Tic-Tac-Toe-q-table-x.txt', "w") as output:
                writer = csv.writer(output, lineterminator = '\n', delimiter = '\t')
                writer.writerows(q_table_x)
            with open('Tic-Tac-Toe-q-table-o.txt', "w") as output:
                writer = csv.writer(output, lineterminator = '\n', delimiter = '\t')
                writer.writerows(q_table_o)
        if (next_state == 0 and (count % check == 0 or count >= episodes)):
            ai_check(states_table, states_random, q_table_x, q_table_o)
    js_python(q_table_x, q_table_o)
    return q_table_x, q_table_o

# Train AI   
q_table_x, q_table_o = q_learning(states_table, states_random, episodes = 75000, save = 100, check = 1000, alpha = 0.2, gamma = 0.9)

# Test AI
ai_check(states_table, states_random, q_table_x, q_table_o)

# View Specific State
state = 101
gs = game_string(states_table[state][1], view_board = True)
