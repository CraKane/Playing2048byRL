import random
import time
import numpy as np
import torch 
from torch.nn import Sequential, Conv2d, Linear, Flatten, Dropout, Softmax, ReLU
from torch.nn.modules import MSELoss
import os 
import collections
import torch.nn as nn
from tensorboardX import SummaryWriter
# from torchsummary import summary
from numba import jit


class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst = [], [], []

        for transition in mini_batch:
            s, a, r = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst)

    def size(self):
        return len(self.buffer)


def learn(model, opt):

    for _ in range(model.train_epochs):
        s, a, r = model.memory.sample(model.batch_size)

        s = s.reshape([-1, 1, model.n_features[0], model.n_features[1]])
        s = model.preprocess_state(s).to('cuda' if model.use_cuda else 'cpu')
        # print(s.dtype)
        pi = torch.zeros((model.batch_size, model.n_actions), dtype=torch.float32).to('cuda' if model.use_cuda else 'cpu')
        a = a.long().to('cuda' if model.use_cuda else 'cpu')
        value = torch.ones((model.batch_size, 1)).to('cuda' if model.use_cuda else 'cpu')
        pi.scatter_(1, a, value).to('cuda' if model.use_cuda else 'cpu')
        # print(pi, a)
        r = r.float().to('cuda' if model.use_cuda else 'cpu')
        # print(r.dtype)
        # print(r.size())

        p, v = model(s)
        # print(p.shape, v.shape)
        # print(p.dtype, v.dtype)

        loss1 = (model.loss(r, v)).float().to('cuda' if model.use_cuda else 'cpu')
        # print(loss1.dtype)
        loss2 = torch.mean(torch.sum(pi*torch.log(p), 1), dtype=torch.float32).to('cuda' if model.use_cuda else 'cpu')  
        # print((pi*torch.log(p)).shape)
        # print(torch.sum(pi*torch.log(p), 1).shape)
        # print(loss2.dtype)
        loss = loss1 - loss2
        print(loss)
        # print(loss.dtype)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.learn_step_counter += 1
    if model.writer:
        model.writer.add_scalar('loss', loss.item(), model.learn_step_counter)

class DeepQNetwork(nn.Module):
    def __init__(self,
                 n_actions,
                 n_features,
                 lr=0.005,
                 memory_size=500,
                 batch_size=32,
                 train_epochs=10,
                 use_cuda=True,
                 last_learn_step = 0,
                 logdir= None,
                 modeldir = 'data',
                 ):
        super(DeepQNetwork,self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.use_cuda=use_cuda
        self.learn_step_counter = last_learn_step
        self.episode = 0
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


    def _build_net_cnn(self):
        self.feature = Sequential(
            Conv2d(1, 16, (3, 3), padding=1),
            ReLU(),
            Conv2d(16, 32, (3, 3), padding=1),
            ReLU(),
            Flatten(),
            Linear(512, 64),
            ReLU(),
        ).to('cuda' if self.use_cuda else 'cpu')
        self.policy = Sequential(
            Linear(64, 4),
            Softmax(dim=1)
        ).to('cuda' if self.use_cuda else 'cpu')
        self.value= Sequential(
            Linear(64, 1),
        ).to('cuda' if self.use_cuda else 'cpu')

        # summary(self.q_eval_model, (1, 4, 4), batch_size=1)
        self.loss = MSELoss()

    def forward(self, s):
        out = self.feature(s)
        return self.policy(out), self.value(out)

    def store_memory(self, s, a, r):
        self.memory.put((s, a, r))
    

    def save_model(self, episode):
        torch.save(self.feature.state_dict(), '{}/2048-feature-{}.h5'.format(self.modeldir, episode))
        torch.save(self.policy.state_dict(), '{}/2048-policy-{}.h5'.format(self.modeldir, episode))
        torch.save(self.value.state_dict(), '{}/2048-value-{}.h5'.format(self.modeldir, episode))
    
    def load_model(self, episode):
        self.feature.load_state_dict(torch.load('{}/2048-feature-{}.h5'.format(self.modeldir, episode), map_location=lambda a, b: a if self.use_cuda==False else None))
        self.policy.load_state_dict(torch.load('{}/2048-policy-{}.h5'.format(self.modeldir, episode), map_location=lambda a, b: a if self.use_cuda==False else None))
        self.value.load_state_dict(torch.load('{}/2048-value-{}.h5'.format(self.modeldir, episode), map_location=lambda a, b: a if self.use_cuda==False else None))