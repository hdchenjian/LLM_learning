import numpy as np
np.set_printoptions(precision=6, suppress=True)

import torch
import torch.nn as nn
#torch.set_grad_enabled(False)

x1 = torch.tensor(1, requires_grad=True, dtype=torch.float)
x2 = torch.tensor(2, requires_grad=True, dtype=torch.float)
x3 = torch.tensor(3, requires_grad=True, dtype=torch.float)

x = torch.tensor([x1, x2, x3])
y = torch.randn(3)

y[0] = x1 * x2 * x3
y[1] = x1 + x2 + x3
y[2] = x1 + x2 * x3
print('x, y', x, y)
y.backward(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float))

print(x1.grad, x2.grad, x3.grad)
#tensor(1.1000) tensor(1.4000) tensor(1.)
exit()

x
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2 + 2
print('y', y)
y.retain_grad()
z = torch.sum(y)
z.backward()
print(y.grad)
print(x.grad)
exit()

x = torch.ones(1, 2, requires_grad=True)
print(x)
y = x**2
print(y)
z = y * y * 3
out = z.mean()
print(z, out)
y.retain_grad()
z.retain_grad()
out.backward()
print('\n', z.grad)
print(y.grad)
print(x.grad)

exit()
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(conv(X).shape, tconv(conv(X)).shape)
print(tconv(conv(X)).shape == X.shape)
