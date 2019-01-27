import gym
from gym import wrappers
import gym_ple
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from PIL import Image

import os

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log') # これでtensorboard
file_names = os.listdir(LOG_DIR)
for file_name in file_names: # logファイルが残っていたら消去
    os.remove(LOG_DIR + '/' + file_name)

writer = SummaryWriter(log_dir = LOG_DIR)

from collections import namedtuple, deque # .framesとかでアクセスできるようにしてる
import random

# 保存する用の型を作成
Transition = namedtuple('Transition', ('frames', 'action', 'next_frames', 'reward'))

class Net(nn.Module):
    """CNN module
    num_actions : int
        number of actions
    """
    def __init__(self, num_actions, input_channel_num=4):
        super(Net,self).__init__()
        # input channel is 4
        # input size = 80 * 80
        self.conv1 = nn.Conv2d(input_channel_num, 32, kernel_size=8, stride=4, padding=122)
        # input size = 80 * 80
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=41)
        # input size = 80 * 80
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # input size = 64 * 80 * 80
        self.fc1 = nn.Linear(64 * 80 * 80, 256)
        # input size = 256
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        # input→conv1→activation(ReLU)
        x = F.relu(self.conv1(x))
        # input→conv2→activation(ReLU)
        x = F.relu(self.conv2(x))
        # input→conv3→activation(ReLU)
        x = F.relu(self.conv3(x))

        # to be flatten array , batch size * 80 * 80
        x = x.view(-1, 64 * 80 * 80)
        # input→fc1→activation(ReLU)
        x = F.relu(self.fc1(x))
        # input→fc2→output
        x = F.relu(self.fc2(x))

        return x

# to add the model to tensorboard
# tensor
dummy_x = Variable(torch.rand(15, 4, 80, 80))
test_model = Net(3)
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

    def push(self, frames, action, next_frames, reward):
        """save the transition to memory
        Parameters
        ------------
        frames : torch.tensor, shape(1 * 4 * 80 * 80)
            the game frame
        action : int
            action number
        next_frames : torch.tensor, shape(1 * 4 * 80 * 80)
            the game frame
        reward : float
        """

        if len(self.memory) < self.capacity: # should save
            self.memory.append(None)
        
        self.memory[self.index] = Transition(frames, action, next_frames, reward)

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
    def __init__(self, num_actions):
        """
        """
        self.num_actions = num_actions # アクションの数、これは環境で得れる

        # メモリを作っておく
        capacity = 50000
        self.memory = ReplayMemory(capacity)

        # バッチサイズ
        self.batch_size = 32

        # 学習率
        self.gamma = 0.99

        self.model = Net(self.num_actions)
        # Fixed Q net
        self._teacher_model = Net(self.num_actions)

        print(self.model) # 確認する
        input()

        # tensorboard用
        self.count = 1

        # 最適化手法
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        """
        """
        if len(self.memory) < self.batch_size :# memory check
            return
        
        transitions = self.memory.sample(self.batch_size) # make mini batch

        # We have
        # Transition * Batchsize
        batch = Transition(*zip(*transitions)) # *zipで、tuple方向の変更、*listで取り出し, turn into [torch.FloatTensor of size 80 * 80 * 4] * BATCH_SIZE, have the name

        # torch.FloatTensor of size BATCH_SIZEx4
        frames_batch = torch.cat(batch.frames) # 1 set is 4 frames
        action_batch = torch.cat(batch.action) # action 
        reward_batch = torch.cat(batch.reward) 
        non_final_next_frames = torch.cat([s for s in batch.next_frames if s is not None])
        
        # estimate mode, Q[s, a] <= Q[s, a] + alpha[R + gamma max_a Q(st+1, at+1) - Q(s, a)]
        
        # calc => Q(s, a)
        self.model.eval()
        # first input, batchsize * (1)
        state_action_values = self.model(frames_batch).gather(1, action_batch) # gather check the note, pick up action_num's value 

        # calc max Q having next frames => gamma max_a Q(st+1, at+1)
        # if not done, check next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_frames)))
        # first all 0
        next_state_values = torch.zeros(self.batch_size)        
        # max(1) => see note, return 
        # detach => pick only tensor but have same storage
        
        next_state_values[non_final_mask] = self._teacher_model(non_final_next_frames).max(1)[0].detach()
        # print("torch.is_storage(obj) = {}".format(torch.is_tensor(self.model(non_final_next_frames).max(1)[0])))
        # print("torch.is_storage(obj) = {}".format(torch.is_tensor(self.model(non_final_next_frames).max(1)[0].detach())))
        # input()

        # calc expected Q(st, at)
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        
        # training mode
        self.model.train()

        # calc loss
        # unsqueeze => [1, 2, 3] => [[1], [2], [3]]
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # save loss
        writer.add_scalar(tag='loss_data', scalar_value=loss.item(), global_step=self.count)
        self.count += 1

        self.optimizer.zero_grad()  # reset grad
        loss.backward()  # backpropagation
        self.optimizer.step()  # update

    def update_teacher_model(self):
        self._teacher_model.load_state_dict(self.model.state_dict())

    def save(self, episode):
        """
        """
        torch.save(self.model, "./param/cnn_{}.pkl".format(episode))

    def decide_action(self, frames, episode):
        """
        only decide the action
        Parameters
        ------------
        frames : torch.tensor, shape(1, 4, 80, 80)
        episode : int
            episode num
        """
        # ε-greedy
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # estimated mode
            with torch.no_grad():
                action = self.model(frames).max(1)[1].view(1, 1) # get the index

        else:
            # random
            action = torch.LongTensor([[random.randrange(self.num_actions)]])  # random

        return action

class Agent():
    def __init__(self, num_actions):
        """
        Parameters
        -----------
        num_states : int
        num_actions : int
        """
        self.brain = DQNNet(num_actions)  # brain 

    def update_q_function(self):
        """
        updating Q function
        """
        self.brain.replay()

    def get_action(self, frames, episode):
        """
        Parameters
        ------------
        frames : torch.tensor, shape(1 * 4 * 80 * 80)
            the game frame
        episode : int
            episode number
        """

        action = self.brain.decide_action(frames, episode)
        return action

    def memorize(self, frames, action, next_frames, reward):
        """
        Parameters
        ------------
        frames : torch.tensor, shape(1 * 4 * 80 * 80)
            the game frame
        action : int
            action number
        next_frames : torch.tensor, shape(1 * 4 * 80 * 80)
            the game frame
        reward : float
        """
        self.brain.memory.push(frames, action, next_frames, reward)

    def update_teacher(self):
        """
        Parameters
        -----------

        """
        self.brain.update_teacher_model()
    
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

        self.agent = Agent(num_actions)
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

            frames = self.observer.init_reset()
            done = False
            total_reward = 0
            
            while not done: # this game does not have end
                if render: # 
                    self.observer.render()
                    self.agent.save(episode)
                
                action = self.agent.get_action(frames, episode)  # get action

                # .item() => to numpy
                next_frames, reward, done = self.observer.step(action.item())

                if done:  
                    next_frames = None  

                # add memory
                self.agent.memorize(frames, action, next_frames, reward)

                # update experience replay
                self.agent.update_q_function()

                # update frames
                frames = next_frames

                # update reward
                total_reward += reward.item()
            
            else:
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

    def __init__(self, env, width, height, num_frame):
        """
        Parameters
        -----------
        env : gym environment
        width : int
        height : int
        num_frame : int
        """
        self.env = env
        self.width = width
        self.height = height
        self.num_frame = num_frame
        self._frames = None

    def init_reset(self):
        """
        initial reset, when the episode starts
        Returns
        ----------
        torch_frames : torch.tensor
        """
        self._frames = deque(maxlen=self.num_frame)
        frame = self.env.reset()
        torch_frames = self._transform(frame)

        return torch_frames

    def step(self, action):
        """
        Parameters
        ------------
        action : int
        Returns
        ----------
        torch_frames : torch.tensor
        reward : torch.tensor
        done : bool
        """
        next_frame, reward, done, _ = self.env.step(action)
        reward = torch.FloatTensor([reward])  # reward
        torch_frames = self._transform(next_frame)

        return torch_frames, reward, done
    
    def render(self):
        """
        """
        self.env.render()

    def _transform(self, frame):
        """
        Parameters
        -------------
        frame : numpy.ndarray
        """
        grayed = Image.fromarray(frame).convert("L") # to gray

        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1

        if len(self._frames) == 0:
            for _ in range(self.num_frame):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)

        np_frames = np.array(self._frames)

        torch_frames = torch.from_numpy(np_frames).type(torch.FloatTensor)  # numpy => torch.tensor
        # print("torch_size = {}".format(torch_frames.size()))
        torch_frames = torch_frames.view(1, self.num_frame, self.width, self.height)
        # print("torch_size = {}".format(torch_frames.size()))
        # input()

        return torch_frames
        
def main():
    """
    """
    env = gym.make('Catcher-v0')
    video_path = "./DQN_video"
    env = wrappers.Monitor(env, video_path, video_callable=(lambda ep: ep % 100 == 0), force=True)

    observer = Observer(env, 80, 80, 4)
    trainer = Trainer(observer)
    trainer.run()

    writer.close()


if __name__ == "__main__":
    main()
