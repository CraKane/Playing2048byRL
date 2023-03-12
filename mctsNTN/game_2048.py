'''
@Author:Drake
2048 游戏环境
存活：+1
消去组合一对数：+10
死局：-5

'''

import numpy as np


HEIGHT = 4
WIDTH = 4

CLEAN_REWARD = 1
SURVIVE_REWARD = 1


def step(s, op, game):
    actions = ['u', 'd', 'l', 'r']
    n_actions = len(actions)
    if op in [0, 1, 2, 3]:
        op_ = actions[op]
    if op_ not in actions:
        return 'op error'
    op_num = np.where(np.array(actions) == op_)[0][0]
    # print(s)
    board = s.copy()
    next_board, clear_score = game._move(s, op_num)
    
    done = False
    reward = clear_score * CLEAN_REWARD
    if game._terminal(next_board):
        reward = -20
        done = True
    elif np.sum(np.abs(next_board - board)) == 0:
        reward = -5

    # print(reward)
    if not done and reward >= 0:
        # print("asdsadasd")
        #     # 随机产生新点
        if np.sum(next_board == 0) == 0:
            done = True
            return next_board, reward, done
        tmp = np.random.choice(a=len(np.where(next_board == 0)[0]), size=1)[0]
        
        xx = np.where(next_board == 0)[0][tmp]
        yy = np.where(next_board == 0)[1][tmp]
        new_score = 2 if np.random.uniform() < 0.9 else 4
        next_board[xx][yy] = new_score
        reward += SURVIVE_REWARD  # 幸存加分

    # print(board, next_board)
    return next_board, reward, done 

class Game2048(object):
    def __init__(self):
        self.actions = ['u', 'd', 'l', 'r']  # 操作：上下左右
        self.n_actions = len(self.actions)
        self.n_features = np.array([HEIGHT, WIDTH])
        self.board = np.zeros(shape=[HEIGHT, WIDTH], dtype=np.int32)
        self.clear_score = 0
        self.reset()

    def reset(self):
        init_places = np.random.choice(a=np.array([5, 6, 9, 10], dtype=np.int32), size=2, replace=False)
        init_digitals = np.random.choice(a=np.array([2, 4], dtype=np.int32), size=2, replace=True)
        self.board = np.zeros(shape=[HEIGHT, WIDTH])
        self.n_step = 0
        self.score = 0
        for i in range(2):
            self.board[init_places[i] // HEIGHT][init_places[i] % WIDTH] = init_digitals[i]

    def step(self, op):
        if op in [0, 1, 2, 3]:
            op = self.actions[op]
        if op not in self.actions:
            return 'op error'
        op_num = np.where(np.array(self.actions) == op)[0][0]
        next_board, clear_score = self._move(self.board.copy(), op_num)
        done = False
        reward = clear_score * CLEAN_REWARD
        if self._terminal(next_board):
            reward = -20
            done = True
        elif np.sum(np.abs(next_board - self.board)) == 0:
            reward = -5

        if not done and reward >= 0:
            #     # 随机产生新点
            if np.sum(next_board == 0) == 0:
                done = True
                return next_board, reward, done
            tmp = np.random.choice(a=len(np.where(next_board == 0)[0]), size=1)[0]
            
            xx = np.where(next_board == 0)[0][tmp]
            yy = np.where(next_board == 0)[1][tmp]
            new_score = 2 if np.random.uniform() < 0.9 else 4
            next_board[xx][yy] = new_score
            self.board = next_board.copy()
            reward += SURVIVE_REWARD  # 幸存加分
            self.n_step += 1
        self.score += reward
        return next_board, reward, done

    def has_score(self, board, action):
        tmp_board, _ = self._move(board.copy(), action)
        return np.sum(np.abs(tmp_board - board))

    def _terminal(self, board):
        for i in range(self.n_actions):
            if self.has_score(board, i):
                return False
        return True

    def _line_squeeze(self, line, inv):
        # push box
        i, j = 3 if inv else 0, 3 if inv else 0
        d = -1 if inv else 1
        while ((j >= 0) if inv else (j < 4)):
            if line[j] != 0:
                t = line[j]
                line[j] = 0
                line[i] = t
                i += d   
            j += d
        return line
    
    def _line_combine(self, line, inv):
        # combine box with same value
        i = 3 if inv else 0
        d = -1 if inv else 1
        while ((i >= 1) if inv else (i < 3)):
            if line[i] != 0 and line[i] == line[i + d]:
                line[i] += line[i + d]
                self.clear_score += line[i]
                line[i + d] = 0
                i += d
            i += d
        return line
        
    def _move(self, board, op_num):
        self.clear_score = 0
        inv = (op_num % 2 == 1) # inverse direction
        f = lambda line: self._line_squeeze(self._line_combine(self._line_squeeze(line, inv), inv), inv)
        for i in range(4):
            if op_num < 2:
                board[:,i] = f(board[:,i])
            else:
                board[i,:] = f(board[i,:])
        return board, self.clear_score

    def get_plat_state(self):
        return self.board.reshape([-1,])

    def get_state(self):
        return self.board.copy()

    def get_score(self):
        return self.score

    def show_game(self):
        print(self.board)

    def play_human(self):

        while(True):
            self.show_game()
            digit = input()
            print('your op:', digit)
            if digit == 'q':
                break
            if digit == 'p':
                self.reset()
            if digit in self.actions:
                self.step(digit)
                print('score:', self.get_score())
