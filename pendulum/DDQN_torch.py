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

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log') # これでtensorboard
file_names = os.listdir(LOG_DIR)
for file_name in file_names: # logファイルが残っていたら消去
    os.remove(LOG_DIR + '/' + file_name)

writer = SummaryWriter(log_dir = LOG_DIR)

from collections import namedtuple, deque # .statesとかでアクセスできるようにしてる
import random

# 保存する用の型を作成
Transition = namedtuple('Transition', ('states', 'action', 'next_states', 'reward'))

class Net(nn.Module):
    """CNN module
    num_actions : int
        number of actions
    """
    def __init__(self, num_states, num_actions):
        """
        Parameters
        -------------
        
        """
        super(Net, self).__init__() # 初期化、スーパークラスのインスタンスを呼び出してる
        self.fc1 = nn.Linear(num_states, 32) # 一層目
        self.fc2 = nn.Linear(32, 32) # 二層目
        self.fc3 = nn.Linear(32, num_actions) # 三層目
    
    def forward(self, x):
        """
        Parameters
        -----------
        x : 
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)

        return output

# to add the model to tensorboard
# tensor
dummy_x = Variable(torch.rand(15, 1, 4))
test_model = Net(4, 2)
writer.add_graph(test_model, (dummy_x, ))

class ReplayMemory():
    """
    Attributes
    -----------

    """
    def __init__(self, capacity):
        """
        """
        self.capacity = capacity # how many do we have stock memory
        self.memory = [] # transition 
        self.index = 0

    def push(self, states, action, next_states, reward):
        """save the transition to memory
        Parameters
        ------------
        states : torch.tensor, shape(1 * 4 * 80 * 80)
            the game state
        action : int
            action number
        next_states : torch.tensor, shape(1 * 4 * 80 * 80)
            the game state
        reward : float
        """

        if len(self.memory) < self.capacity: # should save
            self.memory.append(None)
        
        self.memory[self.index] = Transition(states, action, next_states, reward)

        self.index = (self.index + 1) % self.capacity # index_num start from 0 to self.capacity

    def sample(self, batch_size):
        """
        take the data with random

        Parameters
        ----------
        batch_size : int
            batch size of input data
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        return length of memory
        """
        return len(self.memory)


class DQNNet():
    """
    """
    def __init__(self, num_states, num_actions):
        """
        """
        self.num_actions = num_actions # アクションの数、これは環境で得れる
        self.num_states = num_states

        # メモリを作っておく
        capacity = 50000
        self.memory = ReplayMemory(capacity)

        # バッチサイズ
        self.batch_size = 32
        self.init_memory_size = 200

        # 学習率
        self.gamma = 0.99

        self.model = Net(self.num_states, self.num_actions)
        # Fixed Q net
        self._teacher_model = Net(self.num_states, self.num_actions)

        print(self.model) # 確認する
        input()

        # tensorboard用
        self.count = 1
        
        # policy用
        self.ready_batch = False
        self.epsilon = 0.5

        # 最適化手法
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        """
        """
        if len(self.memory) < self.init_memory_size :# memory check
            return
        
        self.ready_batch = True
        
        transitions = self.memory.sample(self.batch_size) # make mini batch

        # We have
        # Transition * Batchsize
        batch = Transition(*zip(*transitions)) # *zipで、tuple方向の変更、*listで取り出し, turn into [torch.FloatTensor of size 80 * 80 * 4] * BATCH_SIZE, have the name

        # torch.FloatTensor of size BATCH_SIZEx4
        states_batch = torch.cat(batch.states) # 1 set is 4 states
        action_batch = torch.cat(batch.action) # action 
        reward_batch = torch.cat(batch.reward) 
        non_final_next_states = torch.cat([s for s in batch.next_states if s is not None])
        
        # estimate mode, Q[s, a] <= Q[s, a] + alpha[R + gamma max_a Q(st+1, at+1) - Q(s, a)]
        
        # calc => Q(s, a)
        self.model.eval()
        self._teacher_model.eval()

        # first input, batchsize * (1)
        state_action_values = self.model(states_batch).gather(1, action_batch) # gather check the note, pick up action_num's value 

        # calc max Q having next states => gamma max_a Q(st+1, at+1)
        # if not done, check next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_states)))
        # first all 0
        next_state_values = torch.zeros(self.batch_size)
        a_m = torch.zeros(self.batch_size).type(torch.LongTensor)

        # max(1) => see note, return 
        # detach => pick only tensor but have same storage
        # calc main network max a
        a_m[non_final_mask] = self.model(non_final_next_states).detach().max(1)[1] # index

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1) # get max action of main 

        # [minibatch * 1] --> [minibatch]
        next_state_values[non_final_mask] = self._teacher_model(non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
    
        # calc expected Q(st, at)
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        
        # training mode
        self.model.train()

        # calc loss
        # unsqueeze => [1, 2, 3] => [[1], [2], [3]]
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        MSE_loss = nn.MSELoss()
        loss = MSE_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # save loss
        writer.add_scalar(tag='loss_data', scalar_value=loss.item(), global_step=self.count)
        self.count += 1

        self.optimizer.zero_grad()  # reset grad
        loss.backward()  # backpropagation
        clip_grad_norm_(self.model.parameters(), 1.0) # clip
        self.optimizer.step()  # update

    def update_teacher_model(self):
        print("update teacher")
        self._teacher_model.load_state_dict(self.model.state_dict())

    def save(self, episode):
        """
        """
        torch.save(self.model, "./param/nn_{}.pkl".format(episode))

    def decide_action(self, states, episode):
        """
        only decide the action
        Parameters
        ------------
        states : torch.tensor, shape(1, 4, 80, 80)
        episode : int
            episode num
        """
        # ε-greedy
        # epsilon = 0.5 * (1 / (episode + 1))

        if np.random.random() < self.epsilon or not self.ready_batch:
            # random
            action = torch.LongTensor([[random.randrange(self.num_actions)]])  # random
        else:
            # arg max
            self.model.eval()  # estimated mode
            with torch.no_grad():
                action = self.model(states).max(1)[1].view(1, 1) # get the index

        return action

    def update_epsilon(self, MAX_EPISODE):
        """
        Parameters
        -----------
        episode : int
            episode number
        """
        # ε-greedy
        print("update parameters")
        final_epsilon = 1e-3
        initial_epsilon = 0.5
        
        diff = (initial_epsilon - final_epsilon)
        decray = diff / float(MAX_EPISODE)
        self.epsilon = max(self.epsilon-decray, final_epsilon)
        print("epsilon = {}".format(self.epsilon))

class Agent():
    def __init__(self, num_states, num_actions):
        """
        Parameters
        -----------
        num_states : int
        num_actions : int
        """
        self.brain = DQNNet(num_states, num_actions)  # brain 

    def update_q_function(self):
        """
        updating Q function
        """
        self.brain.replay()

    def get_action(self, states, episode):
        """
        Parameters
        ------------
        states : torch.tensor, shape(1 * 4 * 80 * 80)
            the game state
        episode : int
            episode number
        """

        action = self.brain.decide_action(states, episode)
        return action

    def memorize(self, states, action, next_states, reward):
        """
        Parameters
        ------------
        states : torch.tensor, shape(1 * 4 * 80 * 80)
            the game state
        action : int
            action number
        next_states : torch.tensor, shape(1 * 4 * 80 * 80)
            the game state
        reward : float
        """
        self.brain.memory.push(states, action, next_states, reward)

    def update_teacher(self):
        """
        Parameters
        -----------

        """
        self.brain.update_teacher_model()

    def update_parameters(self, MAX_EPISODE):
        """

        """
        self.brain.update_epsilon(MAX_EPISODE)
    
    def save(self, episode):
        """
        """
        self.brain.save(episode)

class Trainer():
    """
    Attributes
    -----------
    env : gym.enviroment
    agent : Agent 
    """
    def __init__(self, observer):
        """
        """
        self.observer = observer

        # XXX: must change
        self.env = observer.env  # game
        num_states = self.env.observation_space.shape[0]  # states num but in this case dont need
        num_actions = self.env.action_space.n  # num action in this case 3

        self.agent = Agent(num_states, num_actions)
        self.agent.update_teacher() # initialize
        self.agent.save(0)

    def run(self, MAX_EPISODE=5000, render=False, report_interval=50):
        """
        Parameters
        ------------
        MAX_EPISODE : int, default is 5000
        render : bool, default is False
        report_interval : int, default is 50
        """

        for episode in range(MAX_EPISODE):
            print("episode {}".format(episode))

            states = self.observer.init_reset()
            done = False
            total_reward = 0
            
            while not done: # this game does not have end
                if render: # 
                    # self.observer.render()
                    self.agent.save(episode)
                
                action = self.agent.get_action(states, episode)  # get action

                # .item() => to numpy
                next_states, reward, done = self.observer.step(action.item())

                if done:  
                    next_states = None  

                # add memory
                self.agent.memorize(states, action, next_states, reward)

                # update experience replay
                self.agent.update_q_function()

                # update states
                states = next_states

                # update reward
                total_reward += reward.item()
            
            else:
                self.agent.update_parameters(MAX_EPISODE)
                if episode % 3 == 0:
                    self.agent.update_teacher()

            # save loss
            writer.add_scalar(tag='reward', scalar_value=total_reward, global_step=episode)

            # report if yes, render and save the path
            if episode % report_interval == 0:
                render = True
            else : 
                render = False


class Observer():
    """
    """

    def __init__(self, env):
        """
        Parameters
        -----------
        env : gym environment
        """
        self.env = env

    def init_reset(self):
        """
        initial reset, when the episode starts
        Returns
        ----------
        torch_states : torch.tensor
        """
        state = self.env.reset()
        state = torch.from_numpy(state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換

        return state

    def step(self, action):
        """
        Parameters
        ------------
        action : int
        Returns
        ----------
        torch_states : torch.tensor
        reward : torch.tensor
        done : bool
        """
        next_state, reward, done, _ = self.env.step(action)
        reward = torch.FloatTensor([reward])  # reward
        next_state = torch.from_numpy(next_state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        next_state = torch.unsqueeze(next_state, 0)  # size 4をsize 1x4に変換

        return next_state, reward, done
    
    def render(self):
        """
        """
        self.env.render()
        
def main():
    """
    """
    env = gym.make('CartPole-v0')
    video_path = "./DQN_video"
    env = wrappers.Monitor(env, video_path, video_callable=(lambda ep: ep % 100 == 0), force=True)

    observer = Observer(env)
    trainer = Trainer(observer)
    trainer.run()

    writer.close()


if __name__ == "__main__":
    main()
