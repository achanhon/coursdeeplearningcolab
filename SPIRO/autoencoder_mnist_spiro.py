import torch


def compute_L1(batchprovider, net):
    with torch.no_grad():
        net.eval()
        net = net.cuda()
        nb, L1 = 0, 0
        for x, _ in batchprovider:
            x = x.cuda()
            z, _ = net(x)

            L1 += (x - z).abs().sum()
            nb += x.shape[0]
    return L1, nb


def training_epoch(batchprovider, net, lr):
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    meanloss = 0
    nb, L1 = 0, 0
    for i, (x, _) in enumerate(batchprovider):
        x = x.cuda()
        z, _ = net(x)

        loss = (x - z).abs().sum()

        with torch.no_grad():
            meanloss += loss.clone().cpu().numpy()
            L1 += loss.clone().cpu().numpy()
            nb += x.shape[0]
            if i % 50 == 49:
                print("loss=", meanloss / 50)
                meanloss = 0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
        optimizer.step()

    return L1, nb


class MyAutoencoder(torch.nn.Module):
    def __init__(self):
        super(MyAutoencoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.l1 = torch.nn.Linear(64 * 7 * 7, 2)

        self.d1 = torch.nn.Linear(2, 512)
        self.d2 = torch.nn.Linear(512, 512)
        self.d3 = torch.nn.Linear(512, 512)
        self.d4 = torch.nn.Linear(512, 512)
        self.d5 = torch.nn.Linear(512, 512)
        self.d6 = torch.nn.Linear(512, 7 * 7 * 64)

        self.tmp = torch.nn.AdaptiveAvgPool2d((28, 28))
        self.convf = torch.nn.Conv2d(64, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        x = torch.nn.functional.leaky_relu(self.conv5(x))

        x = x.view(x.shape[0], 64 * 7 * 7)
        code = torch.nn.functional.sigmoid(self.l1(x) * 100) * 10

        x = torch.nn.functional.leaky_relu(self.d1(code))
        x = torch.nn.functional.leaky_relu(self.d2(x))
        x = torch.nn.functional.leaky_relu(self.d3(x))
        x = torch.nn.functional.leaky_relu(self.d4(x))
        x = torch.nn.functional.leaky_relu(self.d5(x))
        x = torch.nn.functional.leaky_relu(self.d6(x))

        x = x.view(x.shape[0], 64, 7, 7)
        x = self.tmp(x)
        x = self.convf(x)
        x = torch.nn.functional.sigmoid(x * 100)

        return x, code


import torchvision

net = MyAutoencoder()
net = net.cuda()

print("load data")
trainset = torchvision.datasets.MNIST(
    root="build",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
testset = torchvision.datasets.MNIST(
    root="build",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=True, num_workers=2
)

print("train the model on the data")

for epoch in range(8):
    print("epoch", epoch)
    L1, nb = training_epoch(trainloader, net, 0.0001)
    print("train L1", L1 / nb)

print("eval model")
L1, nb = compute_L1(testloader, net)
print("test L1", L1 / nb)

colors = [
    "#fff100",
    "#ff8c00",
    "#e81123",
    "#ec008c",
    "#68217a",
    "#00188f",
    "#00bcf2",
    "#00b294",
    "#009e49",
    "#bad80a",
]

import matplotlib.pyplot as plt

with torch.no_grad():
    for x, y in testloader:
        x = x.cuda()
        _, code = net(x)
        code = code.cpu().numpy()
        code *= 100

        C = [colors[y[i]] for i in range(128)]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(code[:, 0], code[:, 1], c=C)
        plt.show()

        quit()
