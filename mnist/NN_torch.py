# about torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data():
    """
    """
    mnist = fetch_mldata('MNIST original')

    X = mnist.data / 255.
    Y = mnist.target # ラベル

    plt.imshow(X[0].reshape(28, 28), cmap='gray')
    plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/7, random_state=0)

    X_train = torch.Tensor(X_train) # 入力
    X_test = torch.Tensor(X_test)
    Y_train = torch.LongTensor(Y_train) # ラベル、整数だから変える
    Y_test = torch.LongTensor(Y_test)

    dataset_train = TensorDataset(X_train, Y_train)
    dataset_test = TensorDataset(X_test, Y_test) # tensorのデータ・セットになってる

    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True) # batch sizeを決定しておく
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False) 

    return loader_train, loader_test

class SimpleNet(nn.Module):
    """simple 3 layer NN
    Attributes
    ----------

    """
    def __init__(self, n_in, n_mid, n_out):
        """
        Parameters
        -------------
        n_in : int
        n_mid : int
        n_out : int
        """
        super(SimpleNet, self).__init__() # 初期化、スーパークラスのインスタンスを呼び出してる
        self.fc1 = nn.Linear(n_in, n_mid) # 一層目
        self.fc2 = nn.Linear(n_mid, n_mid) # 二層目
        self.fc3 = nn.Linear(n_mid, n_out) # 三層目
    
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


def main():
    # load data
    loader_train, loader_test = load_data()

    # make network
    model = SimpleNet(n_in=28*28*1, n_mid=100, n_out=10)
    
    # loss func
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    iteration_num = 10

    # train する場合
    # trainer mode
    model.train()
    for _ in range(iteration_num):
        for data, target in loader_train: # loaderに入ってる
            optimizer.zero_grad() # 一回勾配を0に
            outputs = model(data) # dataを入れる
            loss = loss_fn(outputs, target) # lossを計算

            loss.backward() # 逆伝播
            optimizer.step() # 重み更新

    print("train done")

    # testモードへ
    model.eval() # 推論モードに切り替え
    correct = 0

    with torch.no_grad(): # 微分しないよの意味

        for data, targets in loader_test:

            outputs = model(data)

            # 推論
            _, predicted = torch.max(outputs.data, 1) # バッチだから何次元目かの話

            correct += predicted.eq(targets.data.view_as(predicted)).sum() # 正解と一緒かどうか
    
    data_num = len(loader_test.dataset)
    print("correct = {}".format(correct.item()/float(data_num)))

if __name__ == "__main__":
    main()