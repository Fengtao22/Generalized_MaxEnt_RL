import numpy as np

class Grid_env:
	def __init__(self, state=(0, 5), DETERMINISTIC = False, BOARD_ROWS = 1, BOARD_COLS = 11):
		self.BOARD_ROWS = BOARD_ROWS
		self.BOARD_COLS = BOARD_COLS
		self.board = np.zeros([self.BOARD_ROWS, self.BOARD_COLS])
		self.state = state
		self.isEnd = False
		self.determine = DETERMINISTIC
		self.mid = BOARD_COLS//2

	def giveReward(self):
		if self.state == (0,0):
			return 100
		elif self.state == (0,self.BOARD_COLS-1):
			return 10
		else:
			return -0.1

	def step(self, action):
		next_state = self.nxtPosition(action)
		reward = self.giveReward()
		self.isEndFunc()
		status = self.isEnd
		self.state = next_state
		return (next_state, reward, status)


	def isEndFunc(self):
		if (self.state == (0,0)) or (self.state == (0,self.BOARD_COLS-1)):
			self.isEnd = True

	def nxtPosition(self, action):
		"""
		action: left, stay, right
		return next position
		"""
		if self.determine:
			if action == "left":
				nxtState = (self.state[0], self.state[1] - 1)
			elif action == "right":
				nxtState = (self.state[0], self.state[1] + 1)
			else:
				nxtState = (self.state[0], self.state[1])
			# if next state legal
			if (nxtState[1] >= 0) and (nxtState[1] <= (self.BOARD_COLS -1)):
				self.board[self.state] = 0   
				return nxtState
			return self.state
		else:
			if action == "left":
				if self.state[1] > self.mid:
					prob = [0.05,0.95]
					res = np.random.choice([1,0], 1, p=prob)
				else:
					if self.state[1] == self.mid or self.state[1] = self.mid-1 
						prob = [0.1, 0.9]
					else:
						prob = [0.5, 0.5]	
					res = np.random.choice([0,1], 1, p=prob) 
				nxtState = (self.state[0], self.state[1] - res)
			elif action == "right":
				if self.state[1] > self.mid:
					prob = [0.05,0.95]
					res = np.random.choice([0,1], 1, p=prob)
				else:
					if self.state[1] == self.mid or self.state[1] = self.mid-1 
						prob = [0.1, 0.9]
					else:
						prob = [0.5, 0.5]
					res = np.random.choice([1,0], 1, p=prob)    
				nxtState = (self.state[0], self.state[1] + res)
			else:
				nxtState = (self.state[0], self.state[1])
			# if next state legal
			if (nxtState[1] >= 0) and (nxtState[1] <= (self.BOARD_COLS -1)):
				self.board[self.state] = 0   
				return nxtState
			return self.state    

	def showBoard(self):
		self.board[self.state] = 1
		for i in range(0, self.BOARD_ROWS):
			print('--------------------------------------------')
			out = '| '
			for j in range(0, self.BOARD_COLS):
				if self.board[i, j] == 1:
					token = '*'
				if self.board[i, j] == 0:
					token = '_'
				if j == 0:
					token = 'G1'
				if j == self.BOARD_COLS-1:
					token = 'G2'	
				out += token + ' | '
			print(out)
		print('--------------------------------------------')
		print('\n')