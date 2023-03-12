import torch
from game_2048 import Game2048
import numpy as np
from tqdm import trange
from model import DeepQNetwork
from plot import plot_reward
import os
from torch.optim import SGD
from model import learn
from mcts import Tree
from mcts import select, simulate, u
score_list = []
EPISODE = 500000
time_simu = 20
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
        while True:
            mct = Tree()
            for j in range(time_simu):
                stmp = s.copy()
                if len(mct.nodes) == 0:
                    mct.add_child(-1, 0, s.copy(), 0, 0)
                    # mct.show()
                    idx = simulate(s.copy(), None, mct, RL, 0, 0, game)      
                else:
                    idx, layer, done, action_index  = select(stmp, mct, game) 
                    
                    if not (done or layer > 10):
                        idx = simulate(mct.nodes[idx], action_index, mct, RL, layer, idx, game)
                        
                mct.bp(idx)

            #UCT
            can_action = [i for i in range(4) if game.has_score(s, i)]
            idxx = [i for i in range(4) if game.has_score(s, i) and i in mct.edgeInfo.keys()]
            # print(idxx)
            fQ = [mct.edgeInfo[i][2] for i in idxx]
            # print(fQ)
            if len(fQ) != 0:
                fQ = np.array(fQ) + np.array(u(0, idxx, mct))
                # print(fQ)
                action_index = idxx[np.argmax(fQ)]
                # print(action_index, idxx)
            else:
                action_index = np.random.choice(can_action) if not len(can_action) == 0 else np.random.randint(0, 3)
                # print(action_index)
        
            s_, r, done = game.step(action_index)
            RL.store_memory(s.reshape([-1, ]), action_index, r)
                

            s = s_
            if done:
                if RL.memory.size() > 1000:
                    learn(RL, opt)
                # print('game:\n', game.board)
                print('max score:', game.get_score(), ' board max:', np.max(game.board), ' play step:', game.n_step)
                score_list.append(game.get_score())
                break
            step += 1
            game_step += 1
        RL.episode = episode + 1
        if (episode + 1) % 1000 == 0:
            if RL.writer:
                RL.writer.add_scalar('avg_reward', np.mean(score_list[-100:]), episode)
        if episode > 10000 and (episode + 1) % 10000 == 0:
            RL.save_model(episode + 1)
            if not os.path.exists("score"):
                os.makedirs("score")
            np.savetxt('score/score_list_{}.txt'.format(episode+1), np.array(score_list[-100:]))

            # print('model saved!')
    print('game over')
    RL.save_model(episode + 1)
    print('model saved!')


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    game = Game2048()
    
    RL = DeepQNetwork(n_actions=game.n_actions,
                      n_features=game.n_features,
                      memory_size=10000,
                      batch_size=128,
                      train_epochs=20,
                      use_cuda=use_cuda,
                      modeldir='model',
                      logdir='log')
    
    opt = SGD(RL.parameters(), lr=RL.lr, weight_decay=0.01)

    run_2048()
    print(score_list)
    np.savetxt('score/score_list.txt', np.array(score_list))
    plot_reward()
