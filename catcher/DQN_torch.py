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
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log') # これでtensorboard
writer = SummaryWriter(log_dir = LOG_DIR)

from collections import namedtuple # .framesとかでアクセスできるようにしてる
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
dummy_x = Variable(torch.rand(13, 1, 80, 80))
test_model = Net(4)
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

    def push(self, frames, action, next_frames, end_flg):
        """save the transition to memory
        """

        if len(self.memory) < self.capacity: # should save
            self.memory.append(None)
        
        self.memory[self.index] = Transition(frames, action, next_frames, end_flg)

        self.index = (self.index + 1) % self.capacity # index_num start from 0 to self.capacity

    def sample(self, batch_size):
        """
        take the data with random 
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

        # メモリを作っておく
        capacity = 50000
        self.memory = ReplayMemory(capacity)

        # バッチサイズ
        self.batch_size = 32

        # 学習率
        self.gamma = 0.99

        self.model = Net(self.num_actions)

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

        batch = Transition(*zip(*transitions)) # *zipで、tuple方向の変更、*listで取り出し

        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がBATCH_SIZE分並んでいるのですが、
        # それを torch.FloatTensor of size BATCH_SIZEx4 に変換
        # 状態(frames) = before 4 frames, action, reward, non_finalの状態のミニバッチのVariableを作成

        frames_batch = torch.cat(batch.frames) # 1 set is 4 frames
        action_batch = torch.cat(batch.action) # action 
        reward_batch = torch.cat(batch.reward) 
        non_final_next_frames = torch.cat([s for s in batch.next_frames if s is not None])
        
        # estimate mode, Q[s, a] = Q[s, a] + alpha[R + gamma max_a Q(st+1, a) - Q(s, a)]
        # calc, Q(s+1, a)
        self.model.eval()

        # first input, batchsize * (1)
        state_action_values = self.model(state_batch).gather(1, action_batch) # gather check the note, pick up action_num's value 

        # if not done, check next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        # first all 0
        next_state_values = torch.zeros(self.batch_size)

        # 次の状態があるindexの最大Q値を求める
        # 出力にアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてそのQ値（index=0）を出力します
        # detachでその値を取り出します
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach() # ちがった、とりあえずこれでストレージを共有して、tensorだけ取り出してるイメージ
        # calc expected Q(st, at)
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # training mode
        self.model.train()

        # 損失関数：smooth_l1_lossはHuberloss
        # expected_state_action_valuesは
        # sizeが[minbatch]になっているので、unsqueezeで[minibatch x 1]へ　常にこの形にする
        # unsqueezeは普通の便利関数
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # lossを保存する
        writer.add_scalar(tag='loss_data', scalar_value=loss.item(), global_step=self.count)
        self.count += 1

        self.optimizer.zero_grad()  # reset grad
        loss.backward()  # backpropagation
        self.optimizer.step()  # update

    def decide_action(self, state, episode):
        """
        ここ自体にはバッチサイズ的な話は入ってこない
        """
        # ε-greedy法で徐々に最適行動を採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # ネットワークを推論
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)は[torch.LongTensor of size 1]　を size 1x1 に変換します

        else:
            # 0,1の行動をランダムに返す
            action = torch.LongTensor([[random.randrange(self.num_actions)]])  # 0,1の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action

class Agent():
    def __init__(self, num_states, num_actions):
        """
        """
        self.brain = DQNNet(num_states, num_actions)  # brain 

    def update_q_function(self):
        """
        updating Q function
        """
        self.brain.replay()

    def get_action(self, state, episode):
        """
        policyにのっとってアクションを取得
        """

        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        """
        記憶する
        """
        self.brain.memory.push(state, action, state_next, reward)

class Environment():
    """
    """
    def __init__(self, env):
        """
        """
        self.env = env  # 環境設定
        num_states = self.env.observation_space.shape[0]  # 環境状態数
        num_actions = self.env.action_space.n  # 環境アクション数
        self.agent = Agent(num_states, num_actions)  # 環境内で行動するAgentを生成

    def run(self, MAX_EPISODE=5000, MAX_STEPS=200, render=False, report_interval=50):
        """

        """

        for episode in range(MAX_EPISODE):  # 最大試行数分繰り返す
            print("episode {}".format(episode))
            observation = self.env.reset()  # 環境の初期化

            state = observation  # 観測をそのまま状態sとして使用
            state = torch.from_numpy(state).type(torch.FloatTensor)  # NumPyをPyTorchのテンソルに変換
            state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換

            for step in range(MAX_STEPS):  # 1エピソードのループ、最大は事前にわかる
                if render: # 描画かどうかの確認
                    self.env.render()

                action = self.agent.get_action(state, episode)  # 行動を求める

                # 行動a_tの実行により、s_{t+1}とdoneフラグを求める
                # actionから.item()を指定して、中身を取り出す
                observation_next, _, done, _ = self.env.step(action.item())  # rewardとinfoは使わないので_にする

                # 報酬clippingなので、さらにepisodeの終了評価と、state_nextを設定する
                if done:  # ステップ数が200経過のみ
                    state_next = None  # 次の状態はないので、Noneを格納
                
                    reward = torch.FloatTensor([1.0])  # 報酬

                    if step > 195:
                        reward = torch.FloatTensor([-1.0])  # stepをマックス使い切ってたら-1
                    else:
                        print("reached!!")

                else:
                    reward = torch.FloatTensor([0.0])  # 普段は報酬0
                    state_next = observation_next  # 観測をそのまま状態とする
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
                    state_next = torch.unsqueeze(state_next, 0)  # size 4をsize 1x4に変換

                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)

                # Experience ReplayでQ関数を更新する
                self.agent.update_q_function()

                # 観測の更新
                state = state_next

                if done:
                    break # 終了した場合

            # report するかどうかの確認
            if episode % report_interval == 0:
                render = True
            else : 
                render = False

    def _transform_frames():
        """
        """
        

def main():
    """
    """
    env = gym.make('Catcher-v0')
    video_path = "./DQN_video"
    env = wrappers.Monitor(env, video_path, video_callable=(lambda ep: ep % 100 == 0), force=True)

    moutain_car = Environment(env)
    moutain_car.run()

    writer.close()


if __name__ == "__main__":
    main()
