import gym
from gym import wrappers
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log') # これでtensorboard
writer = SummaryWriter(log_dir = LOG_DIR)

from collections import namedtuple
import random

# 保存する用の型を作成
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



class ReplayMemory():
    """
    Attributes
    -----------

    """
    def __init__(self, capacity):
        """
        """
        self.capacity = capacity # いくつ保存するか
        self.memory = [] # transition 保存
        self.index = 0 # 保存するindexを示す変数

    def push(self, state, action, state_next, end_flg):
        """save the transition to memory
        """

        if len(self.memory) < self.capacity: # 長さが短い場合
            self.memory.append(None)
        
        self.memory[self.index] = Transition(state, action, state_next, end_flg)

        self.index = (self.index + 1) % self.capacity # 割り算することで一番前に戻る

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
        capacity = 10000
        self.memory = ReplayMemory(capacity)

        # バッチサイズ
        self.batch_size = 32
        self.init_memory_size = 200

        # 学習率
        self.gamma = 0.99

        # ネットワーク構築 keras風
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions)) # action の数だけ出てくる

        # chainer風
        # self.model = SimpleNet()

        print(self.model) # 確認する
        input()

        # tensorboard用
        self.count = 1

        self.ready_batch = False
        self.epsilon = 0.5

        # 最適化手法
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        """
        """
        if len(self.memory) < self.init_memory_size :# メモリの確認、少なかったらリターン
            return

        self.ready_batch = True
        
        transitions = self.memory.sample(self.batch_size) # ミニバッチ作成、持ってくる
        
        # print("transition = {}".format(transitions))
        # input() 

        batch = Transition(*zip(*transitions)) # スターzipで、タプル方向変更、スターlistで取り出し

        # print("batch = {}".format(batch))

        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がBATCH_SIZE分並んでいるのですが、
        # それを torch.FloatTensor of size BATCH_SIZEx4 に変換
        # 状態、行動、報酬、non_finalの状態のミニバッチのVariableを作成

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        
        # ここから推論モード　まず、Q[s, a] = Q[s, a] + alpha[R + gmm max_a Q(st+1, a) - Q(s, a)]
        # のうちQ(s+1, a)をみる
        self.model.eval()

        # まず入力してみる batchsize * (1))？
        state_action_values = self.model(state_batch).gather(1, action_batch) # gatherの挙動はノート参考ね

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))
        # まずは全部0にしておく
        next_state_values = torch.zeros(self.batch_size)

        # 次の状態があるindexの最大Q値を求める
        # 出力にアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてそのQ値（index=0）を出力します
        # detachでその値を取り出します
        # print("before = {}".format(self.model(non_final_next_states).max(1)[0]))
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach() # ちがった、とりあえずこれでストレージを共有して、tensorだけ取り出してるイメージ
        # print("after = {}".format(self.model(non_final_next_states).max(1)[0].detach()))

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # 4.1 ネットワークを訓練モードに切り替える
        self.model.train()

        # 損失関数：smooth_l1_lossはHuberloss
        # expected_state_action_valuesは
        # sizeが[minbatch]になっているので、unsqueezeで[minibatch x 1]へ　常にこの形にする
        # unsqueezeは普通の便利関数
        MSE_loss = nn.MSELoss()
        loss = MSE_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # lossを保存する
        writer.add_scalar(tag='loss_data', scalar_value=loss.item(), global_step=self.count)
        self.count += 1

        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

    def decide_action(self, states, episode):
        """
        ここ自体にはバッチサイズ的な話は入ってこない
        """
        # ε-greedy法で徐々に最適行動を採用する
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

    def update_parameters(self, MAX_EPISODE):
        """

        """
        self.brain.update_epsilon(MAX_EPISODE)

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
                observation_next, reward, done, _ = self.env.step(action.item())  # rewardとinfoは使わないので_にする
                
                """
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
                """
                reward = torch.FloatTensor([reward])  # reward
                state_next = observation_next  # 観測をそのまま状態とする
                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
                state_next = torch.unsqueeze(state_next, 0)  # size 4をsize 1x4に変換換

                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)

                # Experience ReplayでQ関数を更新する
                self.agent.update_q_function()

                # 観測の更新
                state = state_next

                if done:
                    self.agent.update_parameters(MAX_EPISODE)
                    break # 終了した場合

            # report するかどうかの確認
            if episode % report_interval == 0:
                render = True
            else : 
                render = False

def main():
    """
    """
    env = gym.make('MountainCar-v0')
    video_path = "./DQN_video"
    env = wrappers.Monitor(env, video_path, video_callable=(lambda ep: ep % 100 == 0), force=True)

    moutain_car = Environment(env)
    moutain_car.run()

    writer.close()


if __name__ == "__main__":
    main()
