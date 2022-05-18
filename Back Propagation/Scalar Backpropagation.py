import torch

# defining x1, x2, w1, w2
x1 = torch.tensor(-2., requires_grad=True)
x2 = torch.tensor(5., requires_grad=True)
w1 = torch.tensor(-3., requires_grad=True)
w2 = torch.tensor(-1., requires_grad=True)

# calculating y predicted
y_pred = (2 + (torch.sin(x1*w1)**2) + torch.cos(x2*w2))**-1
print("y prediction calculated:", y_pred)

# calculating back propagation gradients
y_pred.backward()

print("Back propagation x1 gradient:", x1.grad)
print("Back propagation w1 gradient:", w1.grad)
print("Back propagation x2 gradient:", x2.grad)
print("Back propagation w2 gradient:", w2.grad)
