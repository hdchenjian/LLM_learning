import numpy as np
np.set_printoptions(precision=6, suppress=True)

import torch
import torch.nn as nn
torch.set_grad_enabled(False)

exit()
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(conv(X).shape, tconv(conv(X)).shape)
print(tconv(conv(X)).shape == X.shape)
