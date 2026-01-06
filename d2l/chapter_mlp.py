import math
import numpy as np
import torch, sys
from torch import nn

#sys.path.insert(0, '/home/user/.bin/learn/train/data/d2l-0.17.6/')
import d2l

def evaluate_loss(net, data_iter, loss):  #@save
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels, true_w, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False)) # 不设置偏置，因为我们已经在多项式中实现了它
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log', xlim=[1, num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            l = (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss))
            animator.add(epoch + 1, l)
            print(epoch + 1, l)
    #true_w = true_w[0:4]
    print('weight:', net[0].weight.data.numpy(), true_w, np.sum(np.abs(net[0].weight.data.numpy() - true_w)))
    d2l.plt.savefig('foo.jpg')

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
#import pdb; pdb.set_trace()
#print('labels', labels, labels.shape)
labels += np.random.normal(scale=0.1, size=labels.shape)

poly_features, labels = [torch.tensor(x, dtype= torch.float32) for x in [poly_features, labels]]

poly_degree = 4
train(poly_features[:n_train, :poly_degree], poly_features[n_train:, :poly_degree], labels[:n_train], labels[n_train:], true_w[0:poly_degree])
