import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

y = w * w * w * x

print(y)

a, b = torch.autograd.grad(outputs=y, inputs=[x, w])

print(a)
print(b)
