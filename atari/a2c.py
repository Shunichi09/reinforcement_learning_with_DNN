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
        layers.append(self.actor_layer)

        self.critic_layer = nn.Linear(256, 1)
        layers.append(self.critic_layer)

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
        # x = F.relu(self.conv3(x))

        # to be flatten array , batch size * 80 * 80
        x = x.view(-1, 32 * 84 * 84)

        x = F.relu(self.fc1(x))

        # actor
        # input→fc1→activation(ReLU)
        actor_output = F.relu(self.actor_layer(x)) # calc action value
        # input→fc2→output
        critic_output = F.relu(self.critic_layer(x)) # calc value of the state

        return  critic_output, actor_output

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
        actions : torch.tensor, shape(batch_size, 1) 
        
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
        self.actor_layer = nn.Linear(32, num_action) # actions
        self.critic_layer = nn.Linear(32, 1) # value 

    def forward(self, x):
        """
        Parameters
        -----------
        x : torch.tensor, shape(num_process * advantage_step, 1, num_states) 
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        actor_output = self.actor_layer(h2)
        critic_output = self.critic_layer(h2)

        return critic_output, actor_output

    def get_action(self, x):
        """
        Parameters
        -----------
        x : torch.tensor, shape(num_process * advantage_step, 1, num_states)

        Returns
        ------------
        action : torch.tensor, shape(NUM_PROCESES, 1)
        """
        value, actor_output = self(x)

        action_probs = F.softmax(actor_output, dim=1) # each row's prob
        action = action_probs.multinomial(num_samples=1) # get samples shape(num_process, 1)

        return action
    
    def get_value(self, x):
        """
        Parameters
        -----------
        x : torch.tensor, shape(num_process * advantage_step, 1, state_num)

        Returns
        ---------
        value : torch.tensor, shape(num_process * advantage_step, 1)
        """

        value, actor_output = self(x)

        return value
    
    def evaluate_actions(self, x, actions):
        """
        Parameters
        ----------
        x : torch.tensor, shape(num_process * advantage_step, 1, state_num)

        actions : torch.tensor, shape(num_process * advantage_step, 1)

        Returns
        --------
        value : torch.tensor, shape()
        action_log_prob : torch.tensor, shape()
        entropy : torch.tensor, shape()
        """
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1) # each row
        action_log_probs = log_probs.gather(1, actions) # get prob

        probs = F.softmax(actor_output, dim=1)  # each row's prob
        entropy = -(log_probs * probs).sum(-1).mean() # entropy is avarage

        return value, action_log_probs, entropy

# to add the model to tensorboard
# tensor
dummy_x = Variable(torch.rand(15, 1, 4))
test_2_model = BasicNN(4, 4)
writer.add_graph(test_2_model, (dummy_x, ))


class RolloutStorage():
    """ instead of Experience replay, we should create the rollout class
    Attributes
    ------------

    """
    def __init__(self, num_steps, num_processes):
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
        self.GAMMA = 0.99

    def insert(self, current_obs, action, reward, mask, NUM_ADVANCED_STEP):
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
            state of None or not None

        Notes
        -------
        We should calculate now states reward
        but we cannot get the reward simultaniously
        次の状態が引数として取られてる
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
        calc the dicount reward with discount rate
        Parameters
        -----------
        next_value : 
        """
        # calc returns (discount reward)
        # reversed range
        self.returns[-1] = next_value # the estimated value (Montecarlo) or decided value, 一番最後に代入
        for ad_step in reversed(range(self.rewards.size(0))): # NUM_STEP, NUM_STEP - 1, ...
            self.returns[ad_step] = self.returns[ad_step + 1] * self.GAMMA *\
                                    self.masks[ad_step + 1] + self.rewards[ad_step] # 1つ前から算出

class Brain(object):
    """
    """

    def __init__(self, NN_model, NUM_ADVANCED_STEP, NUM_PROCESSES):
        """
        Parameters
        ------------
        NN_model : class of NN using pytorch
        """
        self.actor_critic_model = NN_model 
        self.optimizer = optim.Adam(self.actor_critic_model.parameters(), lr=0.01)

        self.NUM_ADVANCED_STEP = NUM_ADVANCED_STEP
        self.NUM_PROCESSES = NUM_PROCESSES


    def update(self, rollouts, max_grad_norm=0.5, value_loss_coef=0.5, entropy_coef=0.01):
        """
        Parameters
        -----------

        """
        

        values, action_log_probs, entropy = self.actor_critic_model.evaluate_actions(
            rollouts.observations[:-1].view(-1, 4),
            rollouts.actions.view(-1, 1)) # input states, action history to NN

        # rollouts.observations[:-1].view(-1, 4) --> torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) --> torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(self.NUM_ADVANCED_STEP, self.NUM_PROCESSES, 1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(self.NUM_ADVANCED_STEP, self.NUM_PROCESSES, 1) # torch.Size([5, 16, 1])

        # calc advantage
        advantages = rollouts.returns[:-1] - values  # torch.Size([5, 16, 1])

        # calc loss of caritic (advantage**2) 
        value_loss = advantages.pow(2).mean()

        # calc actor gain, later we should multiple -1 to turn it loss
        action_gain = (action_log_probs*advantages.detach()).mean()
        # detachしてadvantagesを定数として扱う --> これ自体は更新には必要ないので定数へ

        # total loss
        total_loss = (value_loss * value_loss_coef - action_gain - entropy * entropy_coef)

        # update
        self.actor_critic_model.train()  
        self.optimizer.zero_grad() # initialize  
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic_model.parameters(), max_grad_norm)
        self.optimizer.step() 

class Trainer():
    """
    Attributes
    ------------


    """
    def __init__(self):
        self.ENV = 'CartPole-v0'
        self.NUM_PROCESSES = 16
        self.NUM_ADVANCED_STEP = 5
        

    def train(self, NUM_EPISODES=250, check_learning_interval=3):
        """
        
        """

        # make environment
        envs = [gym.make(self.ENV) for i in range(self.NUM_PROCESSES)]
        test_env = gym.make(self.ENV)
        video_path = "./a2c_video"
        test_env = wrappers.Monitor(test_env, video_path, force=True)
        # envs[0] = test_env

        # make environment
        num_state = envs[0].observation_space.shape[0]  # state num 4
        num_action = envs[0].action_space.n  # action num 2
        actor_critic_model = BasicNN(num_action, num_state)  # make NN
        global_brain = Brain(actor_critic_model, self.NUM_ADVANCED_STEP, self.NUM_PROCESSES)

        # make variables
        current_states = torch.zeros(self.NUM_PROCESSES, num_state)  # torch.Size([16, 4])
        rollouts = RolloutStorage(self.NUM_ADVANCED_STEP, self.NUM_PROCESSES)  # rollouts

        episode_rewards = torch.zeros([self.NUM_PROCESSES, 1])
        final_rewards = torch.zeros([self.NUM_PROCESSES, 1]) 
        states_np = np.zeros([self.NUM_PROCESSES, num_state])
        rewards_np = np.zeros([self.NUM_PROCESSES, 1])  
        dones_np = np.zeros([self.NUM_PROCESSES, 1])  
        each_step = np.zeros(self.NUM_PROCESSES)
        episode = 0

        # initialize, reset returns state
        states = [envs[i].reset() for i in range(self.NUM_PROCESSES)]
        states = np.array(states)
        states = torch.from_numpy(states).float()  # torch.Size([16, 4])
        current_states = states  # current state

        # initialize rollouts
        rollouts.observations[0].copy_(current_states)

        # 実行ループ
        for j in range(NUM_EPISODES * self.NUM_PROCESSES):
            # Advantege TD(5)
            for step in range(self.NUM_ADVANCED_STEP):

                # decide action
                with torch.no_grad(): 
                    actions = actor_critic_model.get_action(rollouts.observations[step])

                # (16,1)→(16,)→tensor-->numpy
                actions_np = actions.squeeze(1).numpy()

                # execute onestep for each process
                for i in range(self.NUM_PROCESSES):
                    states_np[i], rewards_np[i], dones_np[i], _ = envs[i].step(actions_np[i])

                    ## reward clipping
                    # episodeの終了評価と、state_nextを設定
                    if dones_np[i]:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる

                        # 環境0のときのみ出力
                        if i == 0:
                            print('%d Episode: Finished after %d steps' % (
                                episode, each_step[i]+1))
                            episode += 1

                        # set reward
                        if each_step[i] < 195:
                            rewards_np[i] = -1.0  # stands get reward
                        else:
                            rewards_np[i] = 1.0  # if not statnds

                        each_step[i] = 0 # step reset

                        # instead of finish state, we add new states
                        states_np[i] = envs[i].reset() # env reset, new episode

                    else:
                        rewards_np[i] = 0.0  # usually reward is 0
                        each_step[i] += 1

                # reward --> tensor, shape(num_rocess, 1)
                rewards = torch.from_numpy(rewards_np).float()
                episode_rewards += rewards

                # 各実行環境それぞれについて、doneならmaskは0に、継続中ならmaskは1にする
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones_np])

                # 最後の試行の総報酬を更新する
                final_rewards *= masks  # 継続中の場合は1をかけ算してそのまま、done時には0を掛けてリセット
                # 継続中は0を足す、done時にはepisode_rewardsを足す
                final_rewards += (1 - masks) * episode_rewards

                # 試行の総報酬を更新する
                episode_rewards *= masks  # 継続中のmaskは1なのでそのまま、doneの場合は0に

                # all zero if current states are done
                current_states *= masks

                # update current_states 
                states = torch.from_numpy(states_np).float()  # torch.Size([16, 4])
                current_states = states  # current state

                # memorize
                rollouts.insert(current_states, actions.detach(), rewards, masks, self.NUM_ADVANCED_STEP)

            # advanced loop end

            # calc now state's value
            with torch.no_grad():
                next_value = actor_critic_model.get_value(rollouts.observations[-1]).detach()
                # rollouts.observationsのサイズはtorch.Size([6, 16, 4])

            # calc discount reward
            rollouts.compute_returns(next_value)

            # NN update
            global_brain.update(rollouts)

            # reset rollout
            rollouts.after_update()

            # add tensorflow
            writer.add_scalar(tag='total_reward', scalar_value=final_rewards.sum().item(), global_step=j)


        for _ in range(100):
            init_state = test_env.reset()
            state = torch.from_numpy(init_state).float()  # torch.Size([16, 4])
            state = state.view(1, 4)
            done = False
            test_reward = 0

            while not done:
                test_env.render()
                # decide action
                with torch.no_grad():
                    action_tensor = actor_critic_model.get_action(state)

                # (1,1)→(1,)→tensor-->numpy
                action = action_tensor.squeeze(1).numpy()[0]
                # action
                next_state, reward, done, _ = test_env.step(action)

                test_reward += reward

                next_state = torch.from_numpy(next_state).float()
                state = next_state.view(1, 4)
                print("state = {}".format(state))
                print("action = {}".format(action))

            # print(test_reward)
            writer.add_scalar(tag='test_reward', scalar_value=test_reward)

def main():
    
    trainer = Trainer()
    trainer.train()

    writer.close()

if __name__ == "__main__":
    main()

                    


                
