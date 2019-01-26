# reference
## CNN について

https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/

## 引数について

```
self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
```

とかだったら、順番に

- 3枚の画像
- 18枚の画像
- kernel_size = フィルタのサイズ
- stride = どれくらい動かすか、フィルタを
- padding = 外堀の埋め方 default 0

計算方法は、
size = (Input_size - Kernel_size + 2 * padding_size) / Stride_size + 1

## Pooling層

```
torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
```

チャンネル数は変わりません
同じ
計算方法も同じ


## 注意点としては

必ず、全結合層にするときに、reサイズでフラットにする必要性