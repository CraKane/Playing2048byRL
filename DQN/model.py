
import random
import time
import numpy as np
import torch 
from torch.nn import Sequential, Conv2d, Linear, Flatten, Dropout, Softmax, ReLU
from torch.optim import Adam
from torch.nn.modules import MSELoss
import os 
import collections
from tensorboardX import SummaryWriter
# from torchsummary import summary

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1 # leaf node index
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class ReplayBuffer():
    """
        This Memory class is modified based on the original code from:
        https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
        """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    count = 0
    def __init__(self, buffer_limit):
        self.tree = SumTree(buffer_limit)

    def put(self, transition): # When put operator, not only put into the
        transition = np.hstack(transition)
        # print(transition[:env.observation_space.shape[0]])
        max_p = np.max(self.tree.tree[-self.tree.capacity:]) # acquire the max priority of the leaf node
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p, author thinks the newer leaf, the higher priority
        self.count += 1

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / (self.tree.total_p+1e-5)  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / (self.tree.total_p+1e-5)
            ISWeights[i, 0] = np.power((prob+1e-5) / (min_prob+1e-5), -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def size(self):
        return min(self.count, self.tree.capacity)


class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 gameref,
                 lr=0.005,
                 reward_decay=0.9,
                 epsilon=0.1,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 train_epochs=10,
                 epsilon_increment=None,
                 use_cuda=True,
                 last_learn_step = 0,
                 logdir= None,
                 modeldir = 'data',
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay
        self.epsilon_max = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        self.train_epochs = train_epochs
        self.game = gameref
        self.use_cuda=use_cuda
        self.learn_step_counter = last_learn_step
        self.episode = 0
        self.threshold = 10000
        if logdir != None:
            time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
            path = logdir + "/" + time_now
            if not os.path.exists(path):
                os.makedirs(path)
            self.writer = SummaryWriter(path)
        else:
            self.writer = None
        self.modeldir = modeldir
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        #self.learn_step_counter = 0
        self._build_net_cnn()
        # 存储空间下标：0-31:S, 32-63:S_, 64:action, 65:reward
        self.memory = ReplayBuffer(self.memory_size)

    def preprocess_state(self, state):  # 预处理
        return np.log2(state + 1) / 16

    def choose_action(self, state, det=False):
        tstate = state[np.newaxis, np.newaxis, :, :]
        tstate = torch.Tensor(self.preprocess_state(tstate)).to('cuda' if self.use_cuda else 'cpu')
        self.epsilon = min(0.1 + np.exp(self.episode - self.threshold), 0.95)
        # print(self.epsilon)
        if det or np.random.uniform() < self.epsilon:
            action_value = self.q_eval_model(tstate)
            action_value = np.squeeze(action_value.detach().cpu().numpy())
            action_value = [action_value[i] if self.game.has_score(state.reshape([4, 4]), i) else np.min(action_value) - 10 for i in range(4)]
            action_index = np.argmax(action_value)
        else:
            valid_actions = [i for i in range(4) if self.game.has_score(state.reshape([4, 4]), i)]
            # print(valid_actions)
            action_index = random.choice(valid_actions) if not len(valid_actions) == 0 else random.randint(0, 3)
        return action_index

    def _build_net_cnn(self):
        self.q_eval_model = Sequential(
            Conv2d(1, 16, (3, 3), padding=1),
            ReLU(),
            Conv2d(16, 32, (3, 3), padding=1),
            ReLU(),
            Flatten(),
            Linear(512, 64),
            ReLU(),
            Linear(64, 4),
            Softmax(dim=1)
        ).to('cuda' if self.use_cuda else 'cpu')
        self.q_target_model = Sequential(
            Conv2d(1, 16, (3, 3), padding=1),
            ReLU(),
            Conv2d(16, 32, (3, 3), padding=1),
            ReLU(),
            Flatten(),
            Linear(512, 64),
            ReLU(),
            Linear(64, 4),
            Softmax(dim=1)
        ).to('cuda' if self.use_cuda else 'cpu')
        # summary(self.q_eval_model, (1, 4, 4), batch_size=1)
        self.opt = Adam(self.q_eval_model.parameters(), lr=self.lr)
        self.loss = MSELoss()

    def store_memory(self, s, a, r, s_, done_mask):
        self.memory.put((s, a, r, s_, done_mask))


    def target_replace_op(self):
        self.q_target_model.load_state_dict(self.q_eval_model.state_dict())


    def adjust_lr(self, total_reward):
        # print(total_reward)
        lr = 0.005+0.0025*(np.tanh(np.log2(total_reward.cpu())-10)+1)
        return lr

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            # print('target_params_replaced!')

        for _ in range(self.train_epochs):
            b_i, b_m, ISWeights = self.memory.sample(self.batch_size)
            b_m = torch.from_numpy(b_m)
            s, a, r, s_prime, done_mask = b_m[:, :16], b_m[:, 16:16+1], b_m[:, 16+1:16+2], \
                                      b_m[:, -(16+1):-1], b_m[:, -1:]
            # print(s.size(), a.size(), r.size(), s_prime.size(), done_mask.size())

            s = s.reshape([-1, 1, self.n_features[0], self.n_features[1]])
            s = self.preprocess_state(s).to('cuda' if self.use_cuda else 'cpu')
            # print(s.size())
            s_prime = s_prime.reshape([-1, 1, self.n_features[0], self.n_features[1]])
            s_prime = self.preprocess_state(s_prime).to('cuda' if self.use_cuda else 'cpu')
            # print(s_prime.size())
            a = a.long().to('cuda' if self.use_cuda else 'cpu')
            # print(a.size())
            r = r.to('cuda' if self.use_cuda else 'cpu')
            # print(r.size())
            done_mask = done_mask.to('cuda' if self.use_cuda else 'cpu')
            # print(done_mask.size())
            # print(s.size(), a.size(), r.size(), s_prime.size(), done_mask.size())

            max_q_idx = self.q_target_model(s_prime.float()).argmax(1).unsqueeze(1)
            # print(max_q_idx.size())
            q_values = self.q_eval_model(s.float())
            q_a = q_values.gather(1, a)
            # print(q_a.shape)
            max_q_primes = self.q_target_model(s_prime.float())
            # print(max_q_primes.size())
            max_q_prime = max_q_primes.gather(1, max_q_idx)
            target = r + self.gamma * max_q_prime * done_mask
            target = target.to(torch.float32)
            # print(target.shape)
            td_error = torch.abs(target - q_a)
            # print(td_error.shape)
            w = torch.from_numpy(np.sqrt(ISWeights)).to('cuda' if self.use_cuda else 'cpu')
            loss = self.loss(w*q_a, w*target)
            self.memory.batch_update(b_i, td_error.cpu().detach().numpy())
            # print(loss)
            lr = self.adjust_lr(torch.sum(r))
            # print(lr)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        self.learn_step_counter += 1
        if self.writer:
            self.writer.add_scalar('loss', loss.item(), self.learn_step_counter)

    def save_model(self, episode):
        torch.save(self.q_eval_model.state_dict(), '{}/2048-{}.h5'.format(self.modeldir, episode))
    
    def load_model(self, episode):
        self.q_eval_model.load_state_dict(torch.load('{}/2048-{}.h5'.format(self.modeldir, episode), map_location=lambda a, b: a if self.use_cuda==False else None))

    # def load_keras_model(self, km):
    #     self.q_eval_model.load_state_dict(keras_to_pyt(km, self.q_eval_model))
    #     self.q_eval_model = self.q_eval_model.to('cuda' if self.use_cuda else 'cpu')
    #     self.save_model(0)