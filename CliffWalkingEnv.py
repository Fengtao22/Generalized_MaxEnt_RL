#!/usr/bin/env python
# coding: utf-8

# ## The Cliff Walking Environment
# 
# The Cliff Walking environment is a gridworld with a discrete state space and discrete action space. 
# The agent starts at grid cell S. The agent can move (deterministically) to the four neighboring cells 
# by taking actions Up, Down, Left or Right. Trying to move out of the boundary results in staying in 
# the same location. So, for example, trying to move left when at a cell on the leftmost column results 
# in no movement at all and the agent remains in the same location. The agent receives -1 reward per step 
# in most states, and -100 reward when falling off of the cliff. This is an episodic task; 
# termination occurs when the agent reaches the goal grid cell G. Falling off of the cliff results in 
# resetting to the start state, without termination.

# ## Packages.
# 
# We import the following libraries that are required for this assignment. We shall be using the following libraries:
# 1. jdc: Jupyter magic that allows defining classes over multiple jupyter notebook cells.
# 2. numpy: the fundamental package for scientific computing with Python.
# 3. matplotlib: the library for plotting graphs in Python.
# 4. RL-Glue: the library for reinforcement learning experiments.
# 5. BaseEnvironment, BaseAgent: the base classes from which we will inherit when creating the environment and agent classes in order for them to support the RL-Glue framework.
# 6. Manager: the file allowing for visualization and testing.
# 7. itertools.product: the function that can be used easily to compute permutations.
# 8. tqdm.tqdm: Provides progress bars for visualizing the status of loops.
# 
# 
# **NOTE: For this notebook, there is no need to make any calls to methods of random number generators. Spurious or missing calls to random number generators may affect your results.**
### https://github.com/LucasBoTang/Coursera_Reinforcement_Learning/blob/master/02Sample-based_Learning_Methods/01Policy_Evaluation_with_Temporal_Difference_Learning.ipynb

import jdc
import numpy as np
from rl_glue import RLGlue
from Agent import BaseAgent 
from Environment import BaseEnvironment  
from manager import Manager
from itertools import product
from tqdm import tqdm


### env = CliffWalkEnvironment

class CliffWalkEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """
        
        # Note, we can setup the following variables later, in env_start() as it is equivalent. 
        # Code is left here to adhere to the note above, but these variables are initialized once more
        # in env_start() [See the env_start() function below.]
        
        reward = None
        state = None # See Aside
        termination = None
        self.reward_state_term = (reward, state, termination)
        
        # AN ASIDE: Observation is a general term used in the RL-Glue files that can be interachangeably 
        # used with the term "state" for our purposes and for this assignment in particular. 
        # A difference arises in the use of the terms when we have what is called Partial Observability where 
        # the environment may return states that may not fully represent all the information needed to 
        # predict values or make decisions (i.e., the environment is non-Markovian.)
        
        # Set the default height to 4 and width to 12 (as in the diagram given above)
        self.grid_h = env_info.get("grid_height", 4) 
        self.grid_w = env_info.get("grid_width", 12)
        
        # Now, we can define a frame of reference. Let positive x be towards the direction down and 
        # positive y be towards the direction right (following the row-major NumPy convention.)
        # Then, keeping with the usual convention that arrays are 0-indexed, max x is then grid_h - 1 
        # and max y is then grid_w - 1. So, we have:
        # Starting location of agent is the bottom-left corner, (max x, min y). 
        self.start_loc = (self.grid_h - 1, 0)
        # Goal location is the bottom-right corner. (max x, max y).
        self.goal_loc = (self.grid_h - 1, self.grid_w - 1)
        
        # The cliff will contain all the cells between the start_loc and goal_loc.
        self.cliff = [(self.grid_h - 1, i) for i in range(1, (self.grid_w - 1))]
        
        # Take a look at the annotated environment diagram given in the above Jupyter Notebook cell to 
        # verify that your understanding of the above code is correct for the default case, i.e., where 
        # height = 4 and width = 12.

    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state from the environment.
        """
        reward = 0
        # agent_loc will hold the current location of the agent
        self.agent_loc = self.start_loc
        # state is the one dimensional state representation of the agent location.
        state = self.state(self.agent_loc)
        termination = False
        self.reward_state_term = (reward, state, termination)

        return self.reward_state_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """

        if action == 0: # UP (Task 1)
            ### START CODE HERE ###
            # Hint: Look at the code given for the other actions and think about the logic in them.
            possible_next_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            if possible_next_loc[0] >= 0: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
            ### END CODE HERE ###
        elif action == 1: # LEFT
            possible_next_loc = (self.agent_loc[0], self.agent_loc[1] - 1)
            if possible_next_loc[1] >= 0: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
        elif action == 2: # DOWN
            possible_next_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            if possible_next_loc[0] < self.grid_h: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
        elif action == 3: # RIGHT
            possible_next_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
            if possible_next_loc[1] < self.grid_w: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
        else: 
            raise Exception(str(action) + " not in recognized actions [0: Up, 1: Left, 2: Down, 3: Right]!")

        reward = -1
        terminal = False

        if self.agent_loc == self.goal_loc: # Reached Goal!
            terminal = True
        elif self.agent_loc in self.cliff: # Fell into the cliff!
            reward = -100
            self.agent_loc = self.start_loc

        self.reward_state_term = (reward, self.state(self.agent_loc), terminal)
        return self.reward_state_term    

    def env_cleanup(self):
         """Cleanup done after the environment ends"""
        self.agent_loc = self.start_loc
    
    # helper method
    def state(self, loc):
        return loc[0] * self.grid_w + loc[1]