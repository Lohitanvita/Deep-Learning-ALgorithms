import torch
torch.manual_seed(1)

# defining X and W
X = torch.tensor( [[2.], [-1.3], [0.98]], requires_grad=True)
W = torch.tensor( [[-2., -0.3, 1.], [2, -6, -1], [0.88, 0.54, 5]], requires_grad=True)

# calculating y predicted
y_pred = torch.nn.Sigmoid()(torch.matmul(W, X))
print("y prediction calculated:", y_pred)

# calculating loss
losses = (y_pred)**2
avg_loss = losses.sum()/3 # sum of losses by size of y_pred

# calculating back propagation gradients
avg_loss.backward()

print("Back propagation X gradient:", X.grad)
print("Back propagation W gradient:", W.grad)



