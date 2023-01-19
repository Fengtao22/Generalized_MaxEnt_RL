from Grid_env import Grid_env

env = Grid_env(state=(0, 10), DETERMINISTIC = False, BOARD_ROWS = 1, BOARD_COLS = 21)

initial_state = env.state
status = False


cum = 0
gamma = 0.9
step = 0
while not status:
	env.showBoard()
	action = 'left'# 'right'
	next_state, r, status = env.step(action)
	cum += gamma**step *r
	step += 1
	print('next state index: {} reward: {}'.format(next_state[1], r), 'Cum Reward: ', cum)
