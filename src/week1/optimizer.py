# data
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset


x_train = torch.FloatTensor([[73,80,75],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])

y_train = torch.FloatTensor([[152], [185], [196], [142]])

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)


# 모델 초기화
model = MultivariateLinearRegressionModel()
W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = torch.optim.SGD([W,b], lr=1e-5)

np_epochs = 20

for epoch in range(np_epochs+1):
    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.mse_loss(hypothesis, y_train)

    # cost로 H(X)로 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print("Epoch {:4d}/{} hypothesis : {} cost : {:.6f}".format(
        epoch, np_epochs, hypothesis.squeeze().detach(),
        cost.item()
    ))

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93,88,93],
                       [89,91,90],
                       [96,98,100],
                       [73,66,70]
                       ]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y


dataset = CustomDataset()

from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset=dataset,
    batch_size=2, # minibatch의 크기는 통상적으로 2의 제곱으로 설정한다.
    shuffle=True # epoch마다 데이터셋을 섞어서 학습되는 순서를 바꾼다.
)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        #H(x) 계산
        prediction = model(x_train)

        # Cost계산
        cost = F.mse_loss(prediction, y_train)

        # Cost로 H(x)개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader),
            cost.item()
        ))



