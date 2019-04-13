import numpy as np
from PIL import Image
import os

# pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
# tensorboard
from tensorboardX import SummaryWriter
# gym
import gym
from gym import wrappers
import gym_ple

from collections import namedtuple, deque
import random

# tensorboard
LOG_DIR = os.path.join(os.path.dirname(__file__), 'log') # tensorboard
file_names = os.listdir(LOG_DIR) # get the log file
for file_name in file_names: # if there is the log files, remove them!
    os.remove(LOG_DIR + '/' + file_name)

writer = SummaryWriter(log_dir=LOG_DIR)

# GPU or CPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("device is = {}".format(device))

# PARAMETERS
NUM_ACTIONS = 3

# make namedtuple for managing the reward and action data
Transition = namedtuple('Transition', ('frames', 'action', 'next_frames', 'reward'))

class BasicCNN(nn.Module):
    """
    basic cnn module
    Attributes
    -----------
    lay
    """
    def __init__(self, input_channel_num=4):
        super(BasicCNN,self).__init__()

        self.layers = []
        # layer
        self.conv1 = nn.Conv2d(input_channel_num, 32, kernel_size=8, stride=4, padding=122) # input size = 80 * 80
        self.layers.append(self.conv1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=41) # input size = 80 * 80
        self.layers.append(self.conv2)        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # input size = 64 * 80 * 80
        self.layers.append(self.conv3)
        self.fc1 = nn.Linear(64 * 80 * 80, 256) # input size = 256
        self.layers.append(self.fc1)        
        self.fc2 = nn.Linear(256, NUM_ACTIONS)
        self.layers.append(self.fc2)        
        
        # initialize weights
        for layer in self.layers:
            torch.nn.init.normal_(layer.weight, std=0.05)

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
# make tensor
dummy_x = torch.rand(15, 4, 80, 80) # batch * channel * size * size
test_model = BasicCNN()
writer.add_graph(test_model, (dummy_x, ))


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

    def run(self, MAX_EPISODE=1200, render=False, report_interval=50):
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
                self.agent.update_parameters(MAX_EPISODE)
                if episode % 3 == 0:
                    print(episode % 3)
                    self.agent.update_teacher()

            # save reward
            writer.add_scalar(tag='reward', scalar_value=total_reward, global_step=episode)

            # report if yes, render and save the path
            if episode % report_interval == 0:
                render = True
            else : 
                render = False

class Observer():
    """
    Observer of the reinforcement learning
    Attributes
    --------------

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