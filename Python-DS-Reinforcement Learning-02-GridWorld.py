# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: Reinforcement Learning
 
# GitHub Repository: <https://github.com/Valdecy>

############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import random

############################################################################

# Functions: Helper - Plot Environment
def plot_environment(environment, actions, start, size_x = 10, size_y = 10, policy = []):
  values_s   = [index for index, x in np.ndenumerate(environment)]
  keys_s     = list(range(0, environment.shape[0]*environment.shape[1]))
  st_cd_dict = dict(zip(keys_s, values_s))
  m, n       = st_cd_dict[start]
  x, y       = 0, 0
  dy, dx     = 0, 0
  fig        = plt.figure(figsize = [size_x, size_y], facecolor = 'lightgrey')
  ax         = fig.add_subplot(111, xticks = range(environment.shape[1] + 1), yticks = range(environment.shape[0] + 1), position = [0.1, 0.1, 0.8, 0.8])
  plt.gca().invert_yaxis()
  ax.grid(color = 'k', linestyle = '-', linewidth = 1)
  ax.xaxis.set_tick_params(bottom = 'off', top   = 'off', labelbottom = 'off')
  ax.yaxis.set_tick_params(left   = 'off', right = 'off', labelleft   = 'off')
  green_stone = mpatches.Rectangle( (n, m), 1, 1, linewidth = 1, edgecolor = 'k', facecolor = 'lightgreen', clip_on = False)
  ax.add_patch(green_stone)
  if (len(policy) > 0):
    _, policy_cd = policy_path(environment, actions, policy)
  for i in range(0, environment.shape[0]):
    for j in range(0, environment.shape[1]):
      if (len(policy) == 0):
        ax.annotate('st '+ str((environment.shape[1]-1)*i + i + j) , xy = (0.4 + j, 0.55 + i), fontsize = 10, fontweight = 'bold') # + '\n' + '(' + str(i) + ', ' + str(j) + ')'
      if   (environment[i, j] == 0 and len(policy) > 0):
        action = policy_cd[(i, j)]
        if (action == 'Down'):
          x  =  0.5 + j
          y  =  0.5 + i 
          dy =  0.15
          dx =  0
        if (action == 'Up'):
          x  =  0.5 + j
          y  =  0.5 + i 
          dy = -0.15
          dx =  0
        if (action == 'Rigth'):
          x  =  0.5 + j
          y  =  0.5 + i 
          dy =  0
          dx =  0.15
        if (action == 'Left'): 
          x  =  0.5 + j
          y  =  0.5 + i 
          dy =  0
          dx = -0.15
        arrow = mpatches.Arrow(x = x, y = y, dx = dx, dy = dy, width = 0.09, facecolor = 'k', edgecolor = 'k',  clip_on = False, alpha = 1)
        ax.add_patch(arrow )
      elif (environment[i, j] == 1):
        blue_stone = mpatches.Rectangle( (j, i), 1, 1, linewidth = 1, edgecolor = 'k', facecolor = 'lightblue', clip_on = False)
        ax.add_patch(blue_stone)
      elif (environment[i, j] == 2):
        red_stone  = mpatches.Rectangle( (j, i), 1, 1, linewidth = 1, edgecolor = 'k', facecolor = 'orangered', clip_on = False)
        ax.add_patch(red_stone)
      elif (environment[i, j] == 3):
        grey_stone = mpatches.Rectangle( (j, i), 1, 1, linewidth = 1, edgecolor = 'k', facecolor = 'grey',      clip_on = False)
        ax.add_patch(grey_stone)
  return

# Function: Helper - Dictionaries & Tables
def build_dictionaries_tables (environment, actions):
  values_s    = [index for index, x in np.ndenumerate(environment)]
  keys_s      = list(range(0, environment.shape[0]*environment.shape[1]))
  st_cd_dict  = dict(zip(keys_s, values_s)) # Dictionary - (States, Coordinates)
  cd_st_dict  = dict(zip(values_s, keys_s)) # Dictionary - (Coordinates, States)
  keys_a      = list(range(0, len(actions)))
  action_dict = dict(zip(keys_a, actions))  # Dictionary - (index, Actions)
  # Table - States-Action-Next State
  st_ac_nst = np.zeros(shape = (len(keys_s), len(keys_a))) 
  for i in range(0, st_ac_nst.shape[0]):
    for j in range(0, st_ac_nst.shape[1]):
      m, n = st_cd_dict[i]
      if   (action_dict[j] == 'Up'):
        st_ac_nst[i, j] = cd_st_dict[(np.clip(m - 1, 0, environment.shape[0] - 1), n)]
      elif (action_dict[j] == 'Down'):
        st_ac_nst[i, j] = cd_st_dict[(np.clip(m + 1, 0, environment.shape[0] - 1), n)]
      elif (action_dict[j] == 'Left'):
        st_ac_nst[i, j]= cd_st_dict[m, (np.clip(n - 1, 0, environment.shape[1] - 1))]
      elif (action_dict[j] == 'Rigth'):
        st_ac_nst[i, j] = cd_st_dict[m, (np.clip(n + 1, 0, environment.shape[1] - 1))]
  # Table - Q-Table
  path_1  = np.asarray(np.where(environment == 1)).T
  path_2  = np.asarray(np.where(environment == 2)).T
  path_3  = np.asarray(np.where(environment == 3)).T
  q_table = np.random.rand(len(keys_s), len(keys_a))/1000 
  for item in [path_1, path_2, path_3]:
    for i in range(0, item.shape[0]):
      q_table[cd_st_dict[(item[i][0], item[i][1])], :] = 0
  return st_cd_dict, action_dict, st_ac_nst, q_table

# Functions: Helper - Policy Path
def policy_path(environment, actions, policy):
  coordinates, _, next_states, _ = build_dictionaries_tables (environment, actions)
  keys        = list(range(0, policy.shape[0]))
  values      = [next_states[state, np.argmax(policy[state, :])] for state in keys]
  policy_dict = dict(zip(keys, values))
  keys_cd     = [coordinates[states] for states in keys]
  values_cd   = [actions[np.argmax(policy[state, :])] for state in keys]
  policy_cd   = dict(zip(keys_cd, values_cd))
  return policy_dict, policy_cd

############################################################################

# Function: RL - Q-Learning
def q_learning(environment, actions, rewards, learning_rate, discount_factor, epsilon, iterations):
  coordinates, action_dict, next_states, q_table = build_dictionaries_tables (environment, actions)
  path     = np.asarray(np.where(environment == 0)).T
  action   = 0
  r_states = [state for state in  list( range(0, q_table.shape[0] ) ) if list(coordinates[state]) in path.tolist()]
  state    = random.choice(r_states)
  q_upd    = np.array(q_table, copy = True)
  while (iterations > 1):
    rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    if (rand <= epsilon):
      action = random.choice( list( range(0, q_table.shape[1] ) ) )
    else:
      action = np.argmax(q_upd[state, :])
    next_state = int(next_states[state, action])
    i, j       = coordinates[state]
    q_upd[state, action] = (1 - learning_rate)*q_upd[state, action] + (learning_rate)*(rewards[ environment[i, j]  ] + discount_factor*np.amax(q_upd[next_state, :]))
    iterations = iterations - 1
    if (environment[i, j] == 0):
      state = next_state
    else:
      state = random.choice(r_states)
  return  q_upd

############################################################################

# Environment - Locations (0 = Free Path; 1 = Treasure Location; 2 = Fire Location; 3 = Blocked Location)
environment = np.array ([ 
                          [0, 3, 0, 0, 0, 3, 0, 0, 3, 1],
                          [0, 3, 0, 3, 0, 3, 0, 3, 0, 0],
                          [0, 3, 0, 3, 0, 3, 0, 3, 0, 2],
                          [0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
                          [0, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                          [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 3, 0, 3, 3, 3, 3, 3, 3, 3],
                          [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        ])

# Environment - Actions
actions = ['Up', 'Down', 'Left', 'Rigth']

# Environment - Plot
plot_environment(environment, actions, start = 0, size_x = 10, size_y = 10, policy = [])

############################################################################

# Parameters - Algorithm
rewards         = dict(zip([0, 1, 2, 3], [0, 1, -1, -15])) # Reward for Moving = 0, Reward for Treasure = 100, Reward for Fire = -100 and Blocked Path = -100
learning_rate   = 0.20
discount_factor = 0.90
epsilon         = 0.10
iterations      = 75000 

############################################################################

# Optimal Policy
ql_policy = q_learning(environment, actions, rewards, learning_rate, discount_factor, epsilon, iterations)

############################################################################

# Check Policy Table
ql_df = pd.DataFrame(ql_policy, columns = actions)
ql_df.head(n = 5)

############################################################################

# Solution - Plot Policy
plot_environment(environment, actions, start = 0, size_x = 10, size_y = 10, policy = ql_policy)

############################################################################