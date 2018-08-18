from __future__ import print_function
import torch

x = torch.rand(5, 3)
y = torch.rand(5, 3)
#print(x)
#print(x[:, 1])

print(torch.cuda.is_available())


## auto-grad
t = torch.ones(2, 2)
print(t.requires_grad)
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
#print(torch.matmul(y, y))
#print(y*y)
out = z.mean()
print(out)
print(x.grad)
out.backward()
print(x.grad)
