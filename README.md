# Playing2048byRL
## Introduction
- Solving the game 2048 by RL related methods.
- 2048 is 4*4 grid game. Target is combining higher scores like 2048.
![2048](illustration.png)
## Methods
- We try 3 main methods which are good at 2048, DQN, MCTS，NTN.
- DQN Algorithm
![DQN](dqn.png)
- MCTS Algorithm
![MCTS](mcts.png)
- NTN Algorithm, good at chess game
![NTN](ntn.png)
## Results
- The total results are here:
![Total Results](summary.png)
- As for DQN, the results are:
![dqn reward](dqn_reward.png)
- So we can get the maximum tile in dqn is:
![dqn tiles](dqn_tiles.png)
- As for MCTS with DQN, the results are:
![mcts reward](mcts_reward.png)
- So we can get the maximum tile in MCTS with DQN is:
![mcts tiles](mcts_tiles.png)
- As for MCTS with NTN, the results are:
![NTN reward](ntn_reward.png)
- So we can get the maximum tile in MCTS with NTN is:
![NTN tiles](ntn_tiles.png)
## Codes Folder
- DQN for dqn
- mctsDQN for MCTS with DQN
- mctsNTN for MCTS with NTN
## NOTES
- Other details can be seen in "《强化学习理论及应用》课程大作业.pdf"
