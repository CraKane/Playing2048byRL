import os
import time
import numpy as np
from tensorboardX import SummaryWriter



class LUT:
	def __init__(self, modeldir, logdir):
		self.alpha = 0.075
		self.gamma = 0.98
		self.c = 15
		self.episode = 0
		self.learn_step_counter = 0
		self.hori = np.zeros((self.c, self.c, self.c, self.c))
		self.vert = np.zeros((self.c, self.c, self.c, self.c))
		self.circ = np.zeros((self.c, self.c, self.c, self.c, self.c, self.c))

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

	def values(self, s):
		v = 0
		# print(s, len(np.shape(self.circ)), len(np.shape(self.hori)))
		for i in range(len(np.shape(self.circ))):
			if i < len(np.shape(self.hori)):
				v += self.hori[s[i][0]][s[i][1]][s[i][2]][s[i][3]]
				v += self.vert[s[0][i]][s[1][i]][s[2][i]][s[3][i]]
			v += self.circ[s[i//3][i%3]][s[i//3][(i%3)+1]][s[(i//3)+1][i%3]][s[(i//3)+1][(i%3)+1]][s[(i//3)+2][i%3]][s[(i//3)+2][(i%3)+1]]
			# print(i//3, i%3)
			# print(i//3, (i%3)+1)
			# print((i//3)+1, i%3)
			# print((i//3)+1, (i%3)+1)
			# print((i//3)+2, i%3)
			# print((i//3)+2, (i%3)+1)
		return v

	def update(self, s, s_, r, done_mask):
		s = np.log2(s + 1).astype(int)
		s_ = np.log2(s_ + 1).astype(int)
		# print(s, s_)
		new_v_total = r+self.gamma*self.values(s_)*done_mask
		delta_v = new_v_total - self.values(s)
		mean_new_v = self.alpha*delta_v / 3
		for i in range(len(np.shape(self.circ))):
			if i < len(np.shape(self.hori)):
				self.hori[s[i][0]][s[i][1]][s[i][2]][s[i][3]] += mean_new_v
				self.vert[s[0][i]][s[1][i]][s[2][i]][s[3][i]] += mean_new_v
			self.circ[s[i//3][i%3]][s[i//3][(i%3)+1]][s[(i//3)+1][i%3]][s[(i//3)+1][(i%3)+1]][s[(i//3)+2][i%3]][s[(i//3)+2][(i%3)+1]] += mean_new_v
		self.learn_step_counter += 1
		if self.writer:
			self.writer.add_scalar('loss', delta_v, self.learn_step_counter)

	def debug(self):
		s = np.ones((4,4))*64
		print(self.hori.shape, self.vert.shape, self.circ.shape)
		print(self.values(s))

	def load_model(self, episode):
		self.hori = np.load('{}/2048-hori-{}.npy'.format(self.modeldir, episode))
		self.vert = np.load('{}/2048-vert-{}.npy'.format(self.modeldir, episode))
		self.circ = np.load('{}/2048-circ-{}.npy'.format(self.modeldir, episode))

	def save_model(self, episode):
		np.save('{}/2048-hori-{}.npy'.format(self.modeldir, episode), self.hori)
		np.save('{}/2048-vert-{}.npy'.format(self.modeldir, episode), self.vert)
		np.save('{}/2048-circ-{}.npy'.format(self.modeldir, episode), self.circ)


# table = LUT()
# table.debug()