import torch
from game_2048 import Game2048
import numpy as np
from tqdm import trange
from model import DeepQNetwork
from plot import plot_reward
import os
from collections import Counter
score_list = []
max_tiles = []
EPISODE = 500000
SEED = 1212
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def run_2048(load_episode = None):
    step = 0
    if load_episode != None:
        RL.load_model(load_episode)

    for episode in trange(int(EPISODE)):
        # print('episode:', episode)
        game.reset()
        s = game.get_state()
        game_step = 0
        max_tile = 0
        while True:
            action_index = RL.choose_action(s)
            # print(len(s.reshape([-1, ])))
            s_, r, done = game.step(action_index)
            # print('action:', game.actions[action_index])
            # print('game:\n', s_, '\n')

            # Reward Shaping

            # 1. max tile gap
            tmp = np.max(s_)
            r += (tmp - np.max(s))
            done_mask = 0.0 if done else 1.0
            # print(s.reshape([-1, ]).shape)
            # print(s_.reshape([-1, ]).shape)
            RL.store_memory(s.reshape([-1, ]), action_index, r, s_.reshape([-1, ]), done_mask)
            if step > 100 and step % 5 == 0:
                RL.learn()

            s = s_
            max_tile = max(max_tile, tmp)
            if done:
                # print('game:\n', game.board)
                # print('max score:', game.get_score(), ' board sum:', np.sum(game.board), ' play step:', game.n_step)
                max_tiles.append(max_tile)
                score_list.append(game.get_score())
                break
            step += 1
            game_step += 1
        RL.episode = episode + 1
        if (episode + 1) % 100 == 0:
            if RL.writer:
                RL.writer.add_scalar('avg_reward', np.mean(np.array(score_list)[-100:]), episode)
                RL.writer.add_scalar('max_tile', max_tile, episode)
        if episode > 200 and (episode + 1) % 10000 == 0:
            RL.save_model(episode + 1)
            if not os.path.exists("score"):
                os.makedirs("score")
            np.savetxt('score/score_list_{}.txt'.format(episode+1), np.array(score_list))

            # print('model saved!')
    print('game over')
    RL.save_model(episode + 1)
    print('model saved!')


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    game = Game2048()
    
    RL = DeepQNetwork(n_actions=game.n_actions,
                      n_features=game.n_features,
                      gameref=game,
                      memory_size=10000,
                      batch_size=128,
                      train_epochs=10,
                      use_cuda=use_cuda,
                      modeldir='model',
                      logdir='log')
    run_2048()
    print(score_list)
    np.savetxt('score/score_list.txt', np.array(score_list))
    max_tiles = np.array(max_tiles)
    counts = {}
    counter = Counter(max_tiles)
    for i in counter.keys():
        counts[i] = counter[i] / len(max_tiles)
    print(counts)
    plot_reward()
