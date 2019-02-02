import gym
from gym import wrappers
import gym_ple
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from tensorboardX import SummaryWriter

from PIL import Image

import os

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log') # tensorboard
file_names = os.listdir(LOG_DIR)
for file_name in file_names: # if there is log, should remove
    os.remove(LOG_DIR + '/' + file_name)

writer = SummaryWriter(log_dir = LOG_DIR)

from collections import namedtuple, deque
import random

class BasicCNN(nn.Module):
    """CNN module
    Attributes
    -----------

    """
    def __init__(self, num_action, num_input_channel=4, weight_optional_initialization=False):
        """
        Parameters
        -----------
        num_action : int
            number of action
        input channel_num : int
            number of inpurt channel
        weight_optional_initialization : bool
            if you want to initialize the network weight with special prob distribution raise this flag
        
        Notes
        --------
        cnn's padding size
        output = (w - k + 2*p) / s + 1
        """
        super(BasicCNN, self).__init__()    
        layers = []
        # input channel is 4
        # input size = 84 * 84

        self.conv1 = nn.Conv2d(num_input_channel, 16, kernel_size=8, stride=4, padding=128)
        layers.append(self.conv1)
        # input size = 84 * 84
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=43)
        layers.append(self.conv2)
        # input size = 84 * 84
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # layers.append(self.conv3)        
        # input size = 32 * 84 * 84
        self.fc1 = nn.Linear(32 * 84 * 84, 256)
        layers.append(self.fc1) 
        # input size = 256
        self.actor_layer = nn.Linear(256, num_action)
        layers.append(self.actor)

        self.critic_layer = nn.Linear(256, 1)
        layers.append(self.critic)

        # initialize
        if weight_optional_initialization:
            for layer in layers: 
                torch.nn.init.normal_(layer.weight, std=0.05)
        

    def forward(self, x):
        """
        Parameters
        -----------
        x : torch.tensor, shape(batch_size, num_channel, 84, 84) 
        
        Returns
        ---------
        actor_output : torch.tensor, shape(batch_size, 1, action_num)

        critic_output : torch.tensor, shape(batch_size, 1, 1)
        
        """
        # input→conv1→activation(ReLU)
        x = F.relu(self.conv1(x))
        # input→conv2→activation(ReLU)
        x = F.relu(self.conv2(x))
        # input→conv3→activation(ReLU)
        x = F.relu(self.conv3(x))

        # to be flatten array , batch size * 80 * 80
        x = x.view(-1, 32 * 84 * 84)

        # actor
        # input→fc1→activation(ReLU)
        actor_output = F.relu(self.actor_layer(x)) # calc action value
        # input→fc2→output
        critic_output = F.relu(self.critic_layer(x)) # calc value of the state

        return actor_output, critic_output

    def get_action(self, x):
        """
        get the action
        Parameters
        --------------
        x : torch.tensor, shape(batch_size, num_channel, 84, 84) 
        """
        value, actor_output = self(x) # input the network
        # dim=1で行動の種類方向にsoftmaxを計算
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1で行動の種類方向に確率計算
        return action
    
    def get_value(self, x):
        """
        get the value of the state
        Parameters
        --------------
        x : torch.tensor, shape(batch_size, num_channel, 84, 84) 
        """
        value, actor_output = self(x)

        return value
    
    def evaluate_actions(self, x, actions):
        """
        evaluate action (this is the teqnic of a2c)
        Parameters
        -----------
        x : torch.tensor, shape(batch_size, num_channel, 84, 84) 
        actions : torch.tensor, shape(batch_size, 1, 1) 
        
        See Also
        ----------
        add addtional item to the loss equation
        """
        value, actor_output = self(x) # input the network

        log_probs = F.log_softmax(actor_output, dim=1)  # calc log
        # get the probs of the action which agent took 
        action_log_probs = log_probs.gather(1, actions)  # gather -> see note and 1 = dim !

        probs = F.softmax(actor_output, dim=1) # this is not expected value so we calc mean the batch size
        entropy = -(log_probs * probs).sum(-1).mean() # -1 means max dim

        return value, action_log_probs, entropy

# to add the model to tensorboard
# tensor
dummy_x = Variable(torch.rand(15, 4, 84, 84))
test_model = BasicCNN(3)
writer.add_graph(test_model, (dummy_x, ))

class BasicNN(nn.Module):
    """NN module
    num_actions : int
        number of actions
    """
    def __init__(self, num_action, num_state):
        """
        Parameters
        -------------
        num_action : int
            number of action
        num_state : int
            number of state
        """
        super(BasicNN, self).__init__() 
        self.fc1 = nn.Linear(num_state, 32) 
        self.fc2 = nn.Linear(32, 32) 
        self.fc3 = nn.Linear(32, num_action) 

    def forward(self, x):
        """
        Parameters
        -----------
        x : torch.tensor, shape(batch_size, 1, num_states) 
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)

        return output

# to add the model to tensorboard
# tensor
dummy_x = Variable(torch.rand(15, 1, 4))
test_2_model = Net(4, 2)
writer.add_graph(test_2_model, (dummy_x, ))

class RolloutStorage():
    """ instead of Experience replay, we should create the rollout class
    Attributes
    ------------

    """
    def __init__(self, num_steps, num_processes, obs_shape):
        """
        Parameters
        -----------
        num_steps : int
            number of Advantage step
        num_processed : int
            number of agent
        """
        self.observations = torch.zeros(num_steps + 1, num_processes, 4) # 4 is number of state
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        # discount reward
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.index = 0 

    def insert(self, current_obs, action, reward, mask):
        """ insert the transitions to rolloutdtorage
        if you can get new one, that transition add to the front 
        Parameters
        -----------
        current_obs : 
            next state
        action : int
            action
        reward : int
            reward
        mask : int
            state of 

        Notes
        -------
        We should calculate now states reward
        but we cannot get the reward simultaniously
        """
        self.observations[self.index + 1].copy_(current_obs) 
        self.masks[self.index + 1].copy_(mask) # to avoid calc final state
        self.rewards[self.index].copy_(reward) # the reward is 1 step before's state
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # update index

    def after_update(self):
        """
        this is the reset
        so you can add new trainsitions
        """
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        """
        """
        # calc returns (discount reward)
        # reversed range
        self.returns[-1] = next_value # the estimated value (Montecarlo)
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * GAMMA *\
                                    self.masks[ad_step + 1] + self.rewards[ad_step]
2
class Brain(object):
    """
    """

    def __init__(self, NN_model):
        """
        Parameters
        ------------
        NN_model : class of NN using pytorch
        """
        self.actor_critic_model = NN_model 
        self.optimizer = optim.Adam(self.actor_critic_model.parameters(), lr=0.01)

    def update(self, rollouts):
        """

        """
        obs_shape = rollouts.observations.size()[2:]  # torch.Size([4, 84, 84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 4),
            rollouts.actions.view(-1, 1))

        # 注意：各変数のサイズ
        # rollouts.observations[:-1].view(-1, 4) torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(num_steps, num_processes, 1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # advantage（行動価値-状態価値）の計算
        advantages = rollouts.returns[:-1] - values  # torch.Size([5, 16, 1])

        # Criticのlossを計算
        value_loss = advantages.pow(2).mean()

        # Actorのgainを計算、あとでマイナスをかけてlossにする
        action_gain = (action_log_probs*advantages.detach()).mean()
        # detachしてadvantagesを定数として扱う

        # 誤差関数の総和
        total_loss = (value_loss * value_loss_coef -
                      action_gain - entropy * entropy_coef)

        # 結合パラメータを更新
        self.actor_critic.train()  # 訓練モードに
        self.optimizer.zero_grad()  # 勾配をリセット
        total_loss.backward()  # バックプロパゲーションを計算
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        #  一気に結合パラメータが変化しすぎないように、勾配の大きさは最大0.5までにする

        self.optimizer.step()  # 結合パラメータを更新