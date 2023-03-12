from numba import jit
import numpy as np
from game_2048 import step
import torch
import copy
c = 1.414


def u(idx, idxx, tree):
	# P = [tree.edgeInfo[idx*10+i][3] for i in idxx]
	# print(P)
	# P = c*np.array(P)
	# print(P)
	N = [tree.edgeInfo[idx*10+i][0] for i in idxx]
	# print(N)
	N = np.array(N)
	tmp = c*np.sqrt(np.log(1+np.sum(N)) / (1+N))
	# print(tmp)
	return tmp





def select(s, tree, game):
	layer = 0
	idx = 0
	done = False
	action_index = None
	x = copy.deepcopy(tree.nodes)
	while True:
		# UCT
		
		if len(tree.childs[idx].keys()) == 0:
			break
		# print(tree.childs[idx].keys(), idx)
		idxx = [i for i in range(4) if game.has_score(s, i)]
		if len(idxx) == 0:
			break
		# if idx == 2:
		# 	print("select")
		# print(idxx)
		# print(s)	
			# print(tree.edgeInfo.keys())
		fQ = [tree.edgeInfo[idx*10+i][2] for i in idxx]
		# print(fQ)
		fQ = np.array(fQ) + np.array(u(idx, idxx, tree))
		# print(fQ)
		action_index = idxx[np.argmax(fQ)]
		if done or layer > 10:
			break
		# print(tree.childs[idx])
		# print(idx)
		idx = tree.childs[idx][action_index]
		# print("asdasd", idx)
		s = tree.nodes[idx].copy()
		layer += 1
	# print(tree.compare(x))
	
	return idx, layer, done, action_index


def simulate(s, a, tree, model, layer, idx, game):
	done = False
	while True:
		if done or layer > 10:
			break 
		# tree.show()
		# ts = torch.from_numpy(s.reshape([-1, 1, 4, 4])).float()
		# ts = model.preprocess_state(ts).to('cuda' if model.use_cuda else 'cpu')
		# p, v = model(ts)
		idxx = [i for i in range(4) if game.has_score(s, i)] 
		if len(idxx) == 0:
			break
	
		# add children
		ss = []
		# if idx == 2:
		# 	print("simu")
		# print(idxx)
		# print(s)
		for a in idxx:
			s_, r, done = step(s.copy(), a, game)
			s_tmp = s_.copy()
			tree.add_child(idx, a, s_tmp, r)
			ss.append(s_tmp)
			# print(tree.edgeInfo.keys())

		# argmax
		# action_index = p.argmax().item()
		if len(idxx) == 1:
			action_index = 0
		else:
			action_index = np.random.randint(0, len(idxx)-1)
		print(action_index)
		# print(idxx)
		
		# if action_index not in idxx:
		# 	action_index = np.random.choice(idxx)
		
		# print(action_index)

		# p_a = p[0][action_index]
		# print(p_a.item(), v.item())
		idx = tree.childs[idx][idxx[action_index]]
		s = tree.nodes[idx].copy()
		layer += 1
		# tree.show()
	# print(idx)
	return idx



class Tree(object):
	def __init__(self):
		super(Tree, self).__init__()
		self.nodes = []
		self.childs = []
		self.parents = []
		self.values = []
		self.actions = []
		self.edgeInfo = {}

	def size(self):
		return len(self.nodes)

	def root(self):
		return self.nodes[0]

	def get_child(self, idx):
		return self.childs[idx]

	def add_child(self, idx, a, s, r):
		if idx == -1:
			self.nodes.append(s)
			self.childs.append({})
			self.parents.append(-1)
			self.values.append(-1)
			self.actions.append(-1)
			return 0
		else:

			if a in self.childs[idx].keys():
				return self.childs[idx][a]

			self.nodes.append(s)
			self.childs.append({})
			self.childs[idx][a] = len(self.nodes)-1
			self.parents.append(idx)
			v = self.values[idx]+r
			info = [0, v, 0]
			self.edgeInfo[idx*10+a] = info
			# if idx == 2:
				# print(self.edgeInfo.keys())
			self.values.append(v)
			self.actions.append(a)
			return len(self.nodes)-1

	def bp(self, idx):
		now = idx
		while now:
			par = self.parents[now]
			self.edgeInfo[par*10+self.actions[now]][0] += 1
			self.edgeInfo[par*10+self.actions[now]][1] += self.values[now]
			self.edgeInfo[par*10+self.actions[now]][2] = self.edgeInfo[par*10+self.actions[now]][1] / self.edgeInfo[par*10+self.actions[now]][0]
			now = par


	def show(self):
		print(self.nodes[0], 
		self.nodes[1],
		self.nodes[2],
		self.nodes[3],
		self.nodes[4],
		self.nodes[5],
		self.nodes[6],
		self.nodes[7],
		self.nodes[8],
		self.nodes[9],
		self.nodes[10])

	def compare(self, old):
		if len(old) == len(self.nodes):
			for i in range(len(self.nodes)):
				if not (self.nodes[i] == old[i]).all():
					return False
		else:
			l = min(len(old), len(self.nodes))
			l2 = max(len(old), len(self.nodes))
			for i in range(l, l2):
				print(self.nodes[i])
			print(l, l2)
			return False
		return True