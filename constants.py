# -*- coding: utf-8 -*-

import os

LOCAL_T_MAX = 2 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = '/data/deephack/tmp/unreal_checkpoints'
LOG_FILE = '/data/deephack/tmp/unreal_log/unreal_log'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 5e-3   # log_uniform high limit for learning rate
PARALLEL_SIZE = 8 # parallel thread size

# ENV_TYPE = 'lab' # 'lab' or 'gym' or 'maze'
#ENV_NAME = 'seekavoid_arena_01'
# ENV_NAME = 'stairway_to_melon'
#ENV_NAME = 'nav_maze_static_01'

ENV_TYPE = 'gym'
ENV_NAME = 'MsPacman-v0'

PREPARE_SUBMIT = False  # should submit or not
SUBMIT_DIR = "/data/deephack/tmp/gym_submits/"
SUBMIT_VERSION = "sub1"
SUBMIT_OUTPUT = os.path.join(SUBMIT_DIR, "{}_{}".format(ENV_NAME, SUBMIT_VERSION))

INITIAL_ALPHA_LOG_RATE = 0.5 # log_uniform interpolate rate for learning rate
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.001 # entropy regurarlization constant
PIXEL_CHANGE_LAMBDA = 0.1 # 0.01 ~ 0.1 for Lab, 0.0001 ~ 0.01 for Gym
EXPERIENCE_HISTORY_SIZE = 10000 # Experience replay buffer size

USE_PIXEL_CHANGE      = True
USE_VALUE_REPLAY      = True
USE_REWARD_PREDICTION = True

MAX_TIME_STEP = 10 * 10**7
SAVE_INTERVAL_STEP = 100 * 1000

GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True
