from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers

"""
About Q
Q = dict
{state : [ ,  ,  ,  ] } 
"""

def discretize_state(state, env):
    """discretize agent state
    Parameters
    ------------
    state : list
    env : gym class
    Returns
    --------
    position : int
    velocity : int
    """
    state_num = 40
    env_low = env.observation_space.low # 位置と速度の最小値
    env_high = env.observation_space.high #　位置と速度の最大値
    env_dx = (env_high - env_low) / state_num # 40等分
    # 0〜39の離散値に変換する
    position = int((state[0] - env_low[0])/env_dx[0])
    velocity = int((state[1] - env_low[1])/env_dx[1])

    return [position, velocity]

class BaseAgent():
    """Base Agent
    Attributes
    -----------
    Q : dict
    epsilon : float
    reward_log : list
    """
    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    def policy(self, state, actions):
        """epsilon greedy 
        Parameters
        -----------
        state : list
            state of list 
        actions : list
            action list
        """
        # print(self.Q[state[0], state[1]])
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if tuple(state) in self.Q and sum(self.Q[state[0], state[1]]) != 0:
                return np.argmax(self.Q[state[0], state[1]])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                   episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()

class QlearningAgent(BaseAgent):
    """

    """
    def __init__(self, epsilon=0.1):
        """
        """ 
        super().__init__(epsilon)
    
    def learn(self, env, episode_count=10000, gamma=0.99, learning_rate=0.2, render=False, report_interval=100):
        """
        Parameters
        -----------
        env : 


        """
        # step1 initialize
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda : [0] * len(actions))

        # step2 learning
        for e in range(episode_count): # 何エピソード行うか
            print("episode = {}".format(e))
            state = env.reset() # 状態初期化
            state = discretize_state(state, env)
            game_end_flg = False
            
            while not game_end_flg:
                if render: # 描画かどうかの確認
                    env.render()
                
                action = self.policy(state, actions) # epsilon greedy
                next_state, reward, game_end_flg, info = env.step(action) # open AI gym の返り値（観測結果・報酬・ゲーム終了FLG・詳細情報を取得） 
                next_state = discretize_state(next_state, env)

                # update
                gain = reward + gamma * max(self.Q[next_state[0], next_state[1]]) # 次の状態での最大値・これプログラム綺麗
                estimated = self.Q[state[0], state[1]][action] 
                self.Q[state[0], state[1]][action] += learning_rate * (gain - estimated)

                state = next_state

                # print("Q = \n  {}".format(self.Q))
                # input()
            
            else:
                self.log(reward)

            # report するかどうかの確認
            if e % report_interval == 0:
                render = True
            else : 
                render = False

def main():
    """
    """
    env = gym.make('MountainCar-v0')
    video_path = "./Q_video"
    env = wrappers.Monitor(env, video_path, video_callable=(lambda ep: ep % 100 == 0))

    agent = QlearningAgent()
    agent.learn(env)
        
if __name__ == "__main__":
    main()