import torch


# Linear Regressor
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W,b], lr = 0.01)

nb_epochs = 1000
for i in range(1, nb_epochs+1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

# Gradient Decent
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W,b], lr = 0.01)

nb_epochs = 1000
for epoch in range(1, nb_epochs+1):

    #H(x) 계산
    hypothesis = x_train * W + b

    #cost gradient 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)

    print("epoch : {:4d}/ {} Weight : {:3f} Cost : {:6f}".format(epoch,nb_epochs, W.item(), cost.item()))

    # cost H(x)로 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()



