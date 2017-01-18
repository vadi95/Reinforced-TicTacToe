import json
import numpy as np
import time
import random
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense

'''
state - array of len 9 | 1 => O, -1 => X
O always starts first.
'''

display_epoch = 100000  # display after these many epochs / games
stop_epoch = 50000000  # total games to be played

learning_rate = 0.0001  # learning rate of optimizer
max_memory = 25000  # max number of learnings that can be memorized
hidden_size = 500  # number of hidden units in each of the 3 layers
batch_size = 25000  # mini-batch size used to update weights
games_per_batch = 5000  # number of games to play for each minibatch

# reward points
reward_win = 1
reward_loss = -1
reward_draw = 0.8
reward_default = 0


class Tic_Tac_Toe(object):
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        state = self.state
        self.action = action
        cell = int(action)
        if state[cell] == 0:
            if state.count(1) == state.count(-1):
                state[cell] = 1
            else:
                state[cell] = -1
        self.state = state

    def _check_win(self):
        state = self.state
        combinations = [
            # horizontal
            ((0, 0), (1, 0), (2, 0)),
            ((0, 1), (1, 1), (2, 1)),
            ((0, 2), (1, 2), (2, 2)),
            # vertical
            ((0, 0), (0, 1), (0, 2)),
            ((1, 0), (1, 1), (1, 2)),
            ((2, 0), (2, 1), (2, 2)),
            # crossed
            ((0, 0), (1, 1), (2, 2)),
            ((2, 0), (1, 1), (0, 2))
        ]
        for coordinates in combinations:
            letters = [state[3 * x + y] for x, y in coordinates]
            if sum(letters) == 3:
                return 1
            if sum(letters) == -3:
                return 2

        return 0

    def _get_reward(self, player):
        state = self.state
        count_O = state.count(1)
        count_X = state.count(-1)

        if self._check_win() == 1:
            if player == 'O':
                return reward_win
            else:
                return reward_loss

        elif self._check_win() == 2:
            if player == 'O':
                return reward_loss
            else:
                return reward_win

        elif count_O + count_X == 9:
            return reward_draw

        else:
            return reward_default

    def observe(self):
        # return board
        return np.array(self.state).reshape((1, -1))

    def act(self, action, player):
        # update board + get immediate reward ( if any ) + check if game is over
        self._update_state(action)
        reward = self._get_reward(player)
        game_over = reward != 0

        return self.observe(), reward, game_over

    def reset(self):
        # reset board
        self.state = [0 for x in range(9)]


class ExperienceReplay(object):
    def __init__(self, max_memory=10):
        self.max_memory = max_memory
        self.memory = list()

    def remember(self, states):
        self.memory.append([states])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, batch_size=10):
        global learnings
        len_memory = len(self.memory)
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            inputs[i:i + 1] = state_t

            if reward_t > 0:
                targets[i, action_t] = reward_t

        return inputs, targets


def display_board(state):
    print('| ', end="")
    for i in range(3):
        for j in range(3):
            idx = 3 * i + j
            if state[idx] == 1:
                print("O | ", end="")
            elif state[idx] == -1:
                print("X | ", end="")
            else:
                print('  | ', end="")
        print()
        if i != 2:
            print('| ', end="")
    print("-------------")


if __name__ == "__main__":
    start = time.time()

    num_actions = 9
    grid_size = 3

    model1 = Sequential()
    model1.add(Dense(hidden_size, input_shape=(grid_size ** 2,), activation='relu'))
    model1.add(Dense(hidden_size, activation='relu'))
    model1.add(Dense(hidden_size, activation='relu'))
    model1.add(Dense(num_actions, activation='softmax'))
    model1.compile(optimizer=RMSprop(lr=learning_rate),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

    model2 = Sequential()
    model2.add(Dense(hidden_size, input_shape=(grid_size ** 2,), activation='relu'))
    model2.add(Dense(hidden_size, activation='relu'))
    model2.add(Dense(hidden_size, activation='relu'))
    model2.add(Dense(num_actions, activation='softmax'))
    model2.compile(optimizer=RMSprop(lr=learning_rate),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

    # If you want to continue training from a previous model, just uncomment the lines below
    # model1.load_weights("modelO.h5")
    # model2.load_weights("modelX.h5")

    # Define environment/game
    env = Tic_Tac_Toe(grid_size)

    # Initialize experience replay object
    exp_replay_X = ExperienceReplay(max_memory=max_memory)
    exp_replay_O = ExperienceReplay(max_memory=max_memory)

    # Train
    x_win_cnt = 0
    o_win_cnt = 0
    draws = 0
    for e in range(stop_epoch):
        loss_X = 0.
        loss_O = 0.

        env.reset()
        game_over = False
        input_t = env.observe()

        curr_game = []
        reward = 0

        while not game_over:
            input_tm1 = input_t

            possible_actions = list(np.where(input_tm1[0] == 0)[0])

            if env.state.count(1) == env.state.count(-1):
                player = 'O'
            else:
                player = 'X'

            action = random.choice(possible_actions)
            input_t, reward, game_over = env.act(action, player)
            curr_game.append([input_tm1, action, reward, input_t])

            if reward == reward_draw:
                draws += 1
            elif reward == reward_win:
                if player == 'O':
                    o_win_cnt += 1
                else:
                    x_win_cnt += 1

        if reward == reward_draw:
            reward_last_but_one = reward_draw
        else:
            reward_last_but_one = -reward

        # give discounted rewards to previous states
        curr_game[len(curr_game) - 2][2] = reward_last_but_one
        for i in range(len(curr_game) - 3, -1, -2):
            curr_game[i][2] = float(curr_game[i + 2][2]) * 0.8
        for i in range(len(curr_game) - 4, -1, -2):
            curr_game[i][2] = float(curr_game[i + 2][2]) * 0.8

        for i in range(0, len(curr_game), 2):
            exp_replay_O.remember(curr_game[i])

        for i in range(1, len(curr_game), 2):
            exp_replay_X.remember(curr_game[i])

        if e % games_per_batch == games_per_batch - 1:
            if len(exp_replay_O.memory) > 0:
                inputs, targets = exp_replay_O.get_batch(batch_size=batch_size)
                loss_O += model1.train_on_batch(inputs, targets)[0]

            if len(exp_replay_X.memory) > 0:
                inputs, targets = exp_replay_X.get_batch(batch_size=batch_size)
                loss_X += model2.train_on_batch(inputs, targets)[0]
            exp_replay_X.memory = list()
            exp_replay_O.memory = list()

        if e % display_epoch == display_epoch - 1:
            print(
                "Epoch " + str(e) + " / " + str(
                    stop_epoch) + "| Loss_X {:.4f} | Loss_O {:.4f} | X Win count {}| O Win count {} | Draws {}".format(
                    loss_X, loss_O, x_win_cnt, o_win_cnt, draws))
            print("Time : " + str(time.time() - start))
            model1.save_weights("modelO.h5", overwrite=True)
            model2.save_weights("modelX.h5", overwrite=True)
            with open("modelO_full.json", "w") as outfile:
                json.dump(model1.to_json(), outfile)
            with open("modelX_full.json", "w") as outfile:
                json.dump(model2.to_json(), outfile)
