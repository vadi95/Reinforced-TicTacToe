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

For 3x3 TicTacToe there are 5,478 valid and distinct boards
Out of which 958 are terminal states so we don't have to learn actions for these
We can stop training when we have learnt 4520 states
'''

init_train = 100000  # train on only end cases for these many games
display_epoch = 10000  # display after these many epochs / games
stop_epoch = 300000  # total games to be played

learning_rate = 0.00007  # learning rate of optimizer
max_memory = 5  # max number of learnings that can be memorized
hidden_size = 500  # number of hidden units in each of the 3 layers
batch_size = 5  # mini-batch size used to update weights

confidence_threshold = 0.999  # move can be considered confident if probability exceeds this

# reward points
reward_win = 5
reward_loss = -5
reward_draw = 3
reward_default = 0

learnings = 0  # number of distinct states we know how to play
learnt_actions = [[0 for y in range(9)] for x in range(19683)]  # learnt outcomes of actions from each state
learnt_states = [0 for x in range(19683)]  # 19683 (3^9 states)


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

    def get_batch(self, model, model_other, batch_size=10):
        global learnings
        global learnt_actions
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            inputs[i:i + 1] = state_t
            targets[i] = np.zeros(9)

            delete = []

            if reward_t > 0:
                if learnt_states[encode_state(state_t[0])] == 0:
                    learnt_states[encode_state(state_t[0])] = 1
                    learnings += 1
                targets[i, action_t] = 1
            else:
                c1 = 0
                c2 = 0
                for j in range(len(state_tp1[0])):
                    if state_tp1[0][j] == 1:
                        c1 += 1
                    elif state_tp1[0][j] == -1:
                        c2 += 1

                if c1 == c2:
                    player = 'X'
                else:
                    player = 'O'

                temp = Tic_Tac_Toe(grid_size)
                temp.state = list(state_tp1.copy()[0])
                turn = 0
                flag = 0  # checks if we are sure how the game will play out from this state
                value = 0  # end result from current state if both play optimally
                while True:
                    if turn % 2 == 0:
                        q = model_other.predict(np.reshape(temp.state, (1, -1)))
                    else:
                        q = model.predict(np.reshape(temp.state, (1, -1)))

                    if np.amax(q) > confidence_threshold and temp.state[np.argmax(q)] == 0 and \
                            (learnt_actions[encode_state(temp.state)][np.argmax(q)] not in [0, -1] or
                                     learnt_actions[encode_state(temp.state)].count(-1) == 9):
                        state, reward, game_over = temp.act(np.argmax(q), player)
                    else:
                        flag = 1
                        break

                    turn += 1
                    if reward == reward_draw:
                        value = 1
                    elif reward == reward_win:
                        value = 2
                    elif reward == reward_loss:
                        value = -1

                    if game_over:
                        break

                if flag == 0 and value > 0:
                    if value == 1:
                        learnt_actions[encode_state(state_t[0])][action_t] = 2
                        delete.append(i)
                    else:
                        targets[i, action_t] = 1
                        learnt_actions[encode_state(state_t[0])][action_t] = 1
                        for w in range(9):
                            if w != action_t:
                                learnt_actions[encode_state(state_t[0])][w] = -1

                    if learnt_states[encode_state(state_t[0])] == 0:
                        learnt_states[encode_state(state_t[0])] = 1
                        learnings += 1
                else:
                    if flag == 0:
                        if learnt_actions[encode_state(state_t[0])].count(-1) != 8:
                            learnt_actions[encode_state(state_t[0])][action_t] = -1
                        else:
                            learnt_actions[encode_state(state_t[0])][action_t] = 1
                    targets[i, action_t] = 0
                    delete.append(i)

        inputs = np.delete(inputs, delete, 0)
        targets = np.delete(targets, delete, 0)

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


def encode_state(state):
    num = 0
    for i in range(9):
        if state[i] == -1:
            k = 2
        else:
            k = state[i]
        num += k * pow(3, i)
    return num


def setup():
    for i in range(19683):
        j = i
        w = 0
        while j != 0:
            k = j % 3
            if k != 0:
                learnt_actions[i][w] = -1
            w += 1
            j = int(j / 3)


def store_lessons(input_tm1, reward, action, input_t, memory, e):
    global learnt_actions
    global learnings
    global learnt_states
    global init_train
    state = encode_state(input_tm1[0])

    if reward > 0 and learnt_states[state] == 0:
        learnt_states[state] = 1
        learnings += 1

    if learnt_actions[state].count(1) == 0:
        if reward == reward_win:
            learnt_actions[state][action] = 1
            for i in range(9):
                if i != action:
                    learnt_actions[state][i] = -1
        elif reward == reward_draw:
            learnt_actions[state][action] = 2

    if learnt_actions[state].count(1) != 0:
        pos_one = learnt_actions[state].index(1)
        memory.remember([input_tm1, pos_one, reward_win, input_t])
    elif learnt_actions[state].count(0) == 0 and 2 in learnt_actions[state]:
        pos_two = learnt_actions[state].index(2)
        memory.remember([input_tm1, pos_two, reward_draw, input_t])
    elif learnt_actions[state][action] != -1:
        if reward > 0 or e >= init_train:
            memory.remember([input_tm1, action, reward, input_t])


def store_loss_lessons(prev_state, prev_action, memory):
    global learnt_actions
    global learnt_states
    global learnings
    state = encode_state(prev_state[0])

    if learnt_actions[state].count(-1) != 8:
        learnt_actions[state][prev_action] = -1
    else:
        best_action = learnt_actions[state].index(max(learnt_actions[state]))
        learnt_actions[state][best_action] = 1
        res = prev_state.copy()
        res[0][best_action] = 1
        memory.remember([prev_state, best_action, reward_win, res])
        if learnt_states[state] == 0:
            learnt_states[state] = 1
            learnings += 1


if __name__ == "__main__":
    start = time.time()
    setup()

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

        prev_state = input_t
        prev_action = 0
        curr_game = []
        while not game_over:
            input_tm1 = input_t
            possible_actions = []
            if random.random() < 0.1:
                possible_actions = [i for i, e in enumerate(learnt_actions[encode_state(input_tm1[0])]) if e == 0]
            if not possible_actions:
                possible_actions = list(np.where(input_tm1[0] == 0)[0])
            rewardO = 0
            rewardX = 0

            if env.state.count(1) == env.state.count(-1):
                player = 'O'
                memory = exp_replay_O
                other_memory = exp_replay_X
            else:
                player = 'X'
                memory = exp_replay_X
                other_memory = exp_replay_O

            action = random.choice(possible_actions)
            input_t, reward, game_over = env.act(action, player)
            store_lessons(input_tm1, reward, action, input_t, memory, e)

            if reward == reward_win:
                store_loss_lessons(prev_state, prev_action, other_memory)

            prev_state = input_tm1
            prev_action = action

            if reward == reward_draw:
                draws += 1
            elif reward == reward_win:
                if player == 'O':
                    o_win_cnt += 1
                else:
                    x_win_cnt += 1

            if len(exp_replay_O.memory) > 0:
                inputs, targets = exp_replay_O.get_batch(model1, model2, batch_size=batch_size)
                if len(targets) > 0 and np.amax(targets) > 0:
                    loss_O += model1.train_on_batch(inputs, targets)[0]
                exp_replay_O.memory = list()

            if len(exp_replay_X.memory) > 0:
                inputs, targets = exp_replay_X.get_batch(model2, model1, batch_size=batch_size)
                if len(targets) > 0 and np.amax(targets) > 0:
                    loss_X += model2.train_on_batch(inputs, targets)[0]
                exp_replay_X.memory = list()

        if e % display_epoch == display_epoch - 1:
            print("Learnings : " + str(learnings))
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
