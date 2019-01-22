import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

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

        # 最適化手法
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        """
        """
        if len(self.memory) < self.batch_size :# メモリの確認、少なかったらリターン
            return
        
        transitions = self.memory.sample(self.batch_size) # ミニバッチ作成、持ってくる
        
        print("transition = {}".format(transitions))
        input() 

        batch = Transition(*zip(*transitions)) # スターzipで、タプル方向変更、スターlistで取り出し

        print("batch = {}".format(batch))

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
        state_action_values = self.model(state_batch).gather(1, action_batch) # たぶんこれでactionと連結される結果がなので、

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))
        # まずは全部0にしておく
        next_state_values = torch.zeros(BATCH_SIZE)

        # 次の状態があるindexの最大Q値を求める
        # 出力にアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてそのQ値（index=0）を出力します
        # detachでその値を取り出します
        print("before = {}".format(self.model(non_final_next_states).max(1)[0]))
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach() # これ変数じゃなくなるイメージ、tfでいうコンスタント？
        print("after = {}".format(self.model(non_final_next_states).max(1)[0].detach()))

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # 4.1 ネットワークを訓練モードに切り替える
        self.model.train()

        # 損失関数：smooth_l1_lossはHuberloss
        # expected_state_action_valuesは
        # sizeが[minbatch]になっているので、unsqueezeで[minibatch x 1]へ　常にこの形にする
        # unsqueezeは普通の便利関数
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新




def main():
    """
    """
