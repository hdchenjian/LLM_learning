import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),         # 将数据转换成Tensor格式，channel, high, witch,数据在（0， 1）范围内
                                transforms.Normalize(0.5, 0.5)]) # 通过均值和方差将数据归一化到（-1， 1）之间
train_ds = torchvision.datasets.MNIST('data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
imgs, _ = next(iter(dataloader))
print('dataloader iter', len(dataloader), ', batch shape', imgs.shape, ', batch size', dataloader.batch_size, ', sample count', len(dataloader.dataset))

# 输入是长度为 100 的 噪声（正态分布随机数）
# 输出为（1， 28， 28）的图片
class Generator(nn.Module): #创建的 Generator 类继承自 nn.Module
    def __init__(self): # 定义初始化方法
        super(Generator, self).__init__() #继承父类的属性
        self.main = nn.Sequential( #使用Sequential快速创建模型
                                  nn.Linear(100, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 28*28),
                                  nn.Tanh()                     # 输出层使用Tanh()激活函数，使输出-1, 1之间
        )
    def forward(self, x):              # 定义前向传播 x 表示长度为100 的noise输入
        img = self.main(x)
        img = img.view(-1, 28, 28) #将img展平，转化成图片的形式，channel为1可写可不写
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                                  nn.Linear(28*28, 512), #输入是28*28的张量，也就是图片
                                  nn.LeakyReLU(), # 小于0的时候保存一部分梯度
                                  nn.Linear(512, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 1), # 二分类问题，输出到1上
                                  nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.main(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)
loss_fn = torch.nn.BCELoss()

def gen_img_plot(model, epoch):
    test_input = torch.randn(16, 100, device=device)
    #print('test_input', test_input.shape)
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i] + 1)/2) # 确保prediction[i] + 1)/2输出的结果是在0-1之间
        plt.axis('off')
    #plt.show()
    plt.savefig('data/{:0>2d}.jpg'.format(epoch))
    
D_loss = []
G_loss = []
for epoch in range(20):
    d_epoch_loss = 0 # 初始损失值为0
    g_epoch_loss = 0
    for step, (img, _) in enumerate(dataloader): # enumerate加序号
        img = img.to(device) #将数据上传到设备
        size = img.size(0) # 获取每一个批次的大小
        random_noise = torch.randn(size, 100, device=device)  # 随机噪声的大小是size个
        d_optim.zero_grad() # 将判别器前面的梯度归0
        real_output = dis(img)      # 判别器输入真实的图片，real_output是对真实图片的预测结果 
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))      
        d_real_loss.backward() # 求解梯度

        gen_img = gen(random_noise)    
        # 判别器输入生成的图片，fake_output是对生成图片的预测
        # 优化的目标是判别器，对于生成器的参数是不需要做优化的，需要进行梯度阶段，detach()会截断梯度，
        # 得到一个没有梯度的Tensor，这一点很关键
        fake_output = dis(gen_img.detach()) 
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))      
        d_fake_loss.backward() # 求解梯度
        d_loss = d_real_loss + d_fake_loss # 判别器总的损失等于两个损失之和
        d_optim.step() # 进行优化

        g_optim.zero_grad() # 将生成器的所有梯度归0
        fake_output = dis(gen_img) # 将生成器的图片放到判别器中，此时不做截断，因为要优化生成器
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output))      # 生成器的损失
        g_loss.backward() # 计算梯度
        g_optim.step() # 优化
        
        # 将损失累加到定义的数组中，这个过程不需要计算梯度
        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
      
    with torch.no_grad():
        count = len(dataloader)
        d_epoch_loss /= count # 计算平均的loss值
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss.item())
        G_loss.append(g_epoch_loss.item())
        print('Epoch:', epoch, g_epoch_loss, d_epoch_loss)
        gen_img_plot(gen, epoch)
