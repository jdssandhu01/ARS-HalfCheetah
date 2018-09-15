# AI 2018

# Importing the libraries
import os                          # to create an output folder that will have the AI walking video in the pybullet framework
import numpy as np
import gym                         # to connect the env of pybullet to the AI
from gym import wrappers           # to see videos of the AI's walking on the fields in the monitor
import pybullet_envs

# Setting the Hyper Parameters

class Hp():
    
    def __init__(self):                             # self is used to refer to the instance of the class
        self.nb_steps = 1000                        # number of times we will update the model
        self.episode_length = 1000                  # the max time the AI will walk on the field
        self.learning_rate = 0.02                   # to control how fast the AI is learning
        self.nb_directions = 16                     # the number of perturbations to be applied on each of the weights
        self.nb_best_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03                           # the standard deviation (sigma) in the Gaussian distribution that we will use
        self.seed = 1                               # fix the parameters of the environment to have the same results
        self.env_name = 'HalfCheetahBulletEnv-v0'   # the name of the environment

# Normalizing the states

class Normalizer():
    
    def __init__(self, nb_inputs):                              # nb_inputs = number of inputs to the perceptron
        self.n = np.zeros(nb_inputs)                            # a counter vector that keeps track of how many states have already been encountered
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)                    # the numerator of the variance
        self.var = np.zeros(nb_inputs)                          # the variance
    
    def observe(self, x):                                       # this will compute and update the mean and the variance each time we observe a new state; x is the new state
        self.n += 1.                                            # incrementing n as we observe new states
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)   # .clip(1-e2) to make sure the variance is not 0 where 1e-2 = 0.01
    
    def normalize(self, inputs):                                # the normalize method
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)                             # the standard deviation is the root of variance
        return (inputs - obs_mean) / obs_std                    # the normalized value

# Building the AI

class Policy():                                                         # our AI is a policy
    
    def __init__(self, input_size, output_size):                        # self - our future Policy objects or AI's
        self.theta = np.zeros((output_size, input_size))                # self.theta - the matrix of weights of the neurons of the perceptron of the policy
    
    def evaluate(self, input, delta = None, direction = None):          # delta or perturbation is a matrix of small numbers following a normal distribution; delta is None by default if no perturbation is to be applied to the input
        if direction is None:                                           # if no perturbation is applied
            return self.theta.dot(input)                                # returns the action to play
        elif direction == "positive":                                   # positive perturbation
            return (self.theta + hp.noise*delta).dot(input)             # returns the action to play
        else:                                                           # negative perturbation
            return (self.theta - hp.noise*delta).dot(input)             # returns the action to play
    
    def sample_deltas(self):                                                            # will return 16 matrices of small values following a Gaussian distribution of mean 0 and variance 1
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]    # will return the tuple (self.theta.shape[0], self.theta.shape[1])
    
    def update(self, rollouts, sigma_r):                                # approximation of the gradient descent; rollouts is the list of several triplets containing reward in the +ve direction, reward in the -ve direction and the perturbation in that specific direction; sigma_r is the std. of rewards
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d                                 # d is the perturbation in a specific direction
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

# Exploring the policy on one specific direction and over one episode

def explore(env, normalizer, policy, direction = None, delta = None):   # env is the object of the pybullet library
    state = env.reset()                                                 # to get the 1st state
    done = False                                                        # True if we reach the end of an episode
    num_plays = 0.                                                      # the number of actions played
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:                   # till we don't reach the end of an episode
        normalizer.observe(state)                                       # to calc the mean and the variance
        state = normalizer.normalize(state)                             # the normalized state
        action = policy.evaluate(state, delta, direction)               # the action to play in a specific direction
        state, reward, done, _ = env.step(action)                       # the step() method of the pybullet library object env will return the updated state, the reward after completing the action and the done variable that tells whether the episode is over or not
        reward = max(min(reward, 1), -1)                                # will set large positive and large negative rewards as 1 and -1 respectively
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

# Training the AI

def train(env, policy, normalizer, hp):
    
    for step in range(hp.nb_steps):
        
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        
        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])       # getting the positive rewards for the kth direction using the kth perturbation
        
        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])       # getting the negative rewards for the kth direction using the kth perturbation
        
        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}    # will return a dictionary containing the keys - 0 to 15 for directions and values - the max of the +ve and -ve rewards; zip() to gather the +ve and the -ve rewards
        order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:hp.nb_best_directions]             # list of sorted keys corresponding to the best rewards for the best directions
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]                           # setting the rollouts - the best direction triplets (r_pos, r_neg, d)
        
        # Updating our policy
        policy.update(rollouts, sigma_r)
        
        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)            # total accumulated rewards without any direction and without any perturbation
        print('Step:', step, 'Reward:', reward_evaluation)

# Running the main code

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()                                                       # to have the same results
np.random.seed(hp.seed)
env = gym.make(hp.env_name)                                     # to create the env
env = wrappers.Monitor(env, monitor_dir, force = True)          # to save the videos in the monitor_dir; force = True to prevent any warnings to stop the training
nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
policy = Policy(nb_inputs, nb_outputs)                          # initialized our AI, i.e our perceptron with weights = 0
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)                              # calling the training function
