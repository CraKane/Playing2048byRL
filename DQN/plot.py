import matplotlib.pyplot as plt
import numpy as np




def plot_reward():
	rewards = np.loadtxt('score_list.txt')
	print(rewards)
	plt.figure()
	mean_rewards = np.convolve(rewards, np.ones(100)/100, mode='same')
	print(mean_rewards)
	x = range(len(mean_rewards))
	plt.plot(x,mean_rewards)
	plt.xlabel('episode')
	plt.ylabel('mean_reward')
	plt.title("reward in all episode")
	plt.show()



if __name__ == "__main__":
	plot_reward()