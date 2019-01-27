# torchの使い方

# cat
結合系

```
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
```

# detau()

variableからtensorにする
gradのメソッドがなくなる
たぶんだけど
NNからの出力には変数傾きがついてるっぽい
tensorにしてる感じ

このケースで言えば普通に、配列の中にいれられないから的な

# MAX

```
>>> a = torch.randn(4, 4)
>>> a
tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
        [ 1.1949, -1.1127, -2.2379, -0.6702],
        [ 1.5717, -0.9207,  0.1297, -1.8768],
        [-0.6172,  1.0036, -0.6060, -0.2432]])
>>> torch.max(a, 1)
(tensor([ 0.8475,  1.1949,  1.5717,  1.0036]), tensor([ 3,  0,  0,  1]))
```

になるらしい
返り値はtuple
tupleなのケア

# Storage

1次元配列らしい

# gather

うしろが何番目をとってくるのかの指示になっている
このexampleの使い方しないかも
基本的には何番目的な感じかと

```
>>> t = torch.tensor([[1,2],[3,4]])
>>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
tensor([[ 1,  1],
        [ 4,  3]])
```


# view
reshapeと同じ

```
>>> x = torch.randn(4, 4)
>>> x.size()
torch.Size([4, 4])
>>> y = x.view(16)
>>> y.size()
torch.Size([16])
>>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
>>> z.size()
torch.Size([2, 8])
```


# tensor.item()

floatを取り出してる

```
>>> x = torch.tensor([[1]])
>>> x
tensor([[ 1]])
>>> x.item()
1
>>> x = torch.tensor(2.5)
>>> x
tensor(2.5000)
>>> x.item()
2.5
```

# 重み保存

```
torch.save(the_model.state_dict(), PATH)
```

load

```
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```

# 重み初期化

```
self.conv1 = nn.Conv2d(input_channel_num, 32, kernel_size=8, stride=4, padding=122)
torch.nn.init.normal_(self.conv1.weight, std=0.05)
```

こんな感じで初期化可能

# gpuについて


```
device = torch.device("cuda")

self.model.to(device)

.to('cpu')
```

基本的にはエラーが出てくれるから大丈夫だが

入力、出力はGPUになってるのでこっちで宣言したもの使いたい場合は元に戻さないといけない
もしすべてGPUにしたいならそれでも可な気がする
そこはどっちでもいいんだと思う

# 性質

torch.tensor * 2.(float)
は可能っぽい