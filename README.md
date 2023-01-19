# Generalized_MaxEnt_RL
This is the implementation for the paper of Generalized Maximum Entropy Reinforcement Learning (SSPG)

#### Set up env: (https://www.chenshiyu.top/blog/2019/06/19/Tutorial-Installation-and-Configuration-of-MuJoCo-Gym-Baselines/)
'''
conda create -n mujoco-gym python=3.6
conda activate mujoco-gym
'''

cd /locations_you_want
'''
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -e .
'''

#### install Gym (https://github.com/openai/mujoco-py/issues/477)
Download MuJoCo and its licence from https://www.roboti.us/download.html

Extract amd move the file to the specified path (also add path to .bashrc)

Following the format:
export LD_LIBRARY_PATH=/home/csy/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

'''
pip install gym[all]==0.15.3

May needs: sudo apt-get install libosmesa6-dev (https://github.com/ethz-asl/reinmav-gym/issues/35)
'''


#### Install Pytorch:
'''
conda install -c pytorch pytorch

'''


#### Test env
'''
python gym_test.py
'''
##### MuJoCo is deterministic env

## Create testing env (Grid)


