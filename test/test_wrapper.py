import torch
from practicegrad.wrapper import Wrapper

x = Wrapper(4.0)
z = 2 * x + 2 + x
q = z + z * x
h = (z * z).tanh()
y = h + q + q * x
y.backward()
xmg, ymg = x, y

x1 = torch.Tensor([4.0]).double()
x1.requires_grad = True
z1 = 2 * x1 + 2 + x1
q1 = z1 + z1 * x1
h1 = (z1 * z1).tanh()
y1 = h1 + q1 + q1 * x1
y1.backward()
xpt, ypt = x1, y1

# print(z.grad,z1.grad.item())
# print(q.grad,q1.grad.item())
# print(h.grad,h1.grad.item())
# print(y.grad,y1.grad.item())
print(x)
print(xmg.grad, xpt.grad.item())
# forward pass went well
assert ymg.val == ypt.data.item()
# backward pass went well
assert xmg.grad == xpt.grad.item()
