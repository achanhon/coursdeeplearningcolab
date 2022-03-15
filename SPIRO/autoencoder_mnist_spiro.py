import torch


def compute_L1(batchprovider, net):
    with torch.no_grad():
        net.eval()
        net = net.cuda()
        nb, L1 = 0, 0
        for x, _ in batchprovider:
            x = x.cuda()
            z = net(x)

            L1 += (x - z).abs().sum()
            nb += x.shape[0]
    return L1, nb


def training_epoch(batchprovider, net, lr):
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    meanloss = 0
    for i, (x, _) in enumerate(batchprovider):
        x = x.cuda()
        z, _ = net(x)

        loss = (x - z).abs().sum()

        with torch.no_grad():
            meanloss += loss.clone().cpu().numpy()
            if i % 50 == 49:
                print("loss=", meanloss / 50)
                meanloss = 0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
        optimizer.step()


class MyAutoencoder(torch.nn.Module):
    def __init__(self):
        super(MyAutoencoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16)
        self.conv2 = torch.nn.Conv2d(16, 64)
        self.l1 = torch.nn.Linear(64 * 8 * 8, 2)

        self.d1 = torch.nn.Linear(2, 512)
        self.d2 = torch.nn.Linear(512, 512)
        self.d3 = torch.nn.Linear(512, 8 * 8 * 64)
        self.convd1 = torch.nn.Conv2d(64, 16)
        self.convd2 = torch.nn.Conv2d(16, 1)

        self.tmp = torch.nn.AdaptiveAvgPool2d((16, 16))
        self.tmp2 = torch.nn.AdaptiveAvgPool2d((32, 32))

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = x.view(x.shape[0], 64 * 8 * 8)

        code = self.nn.functional.sigmoid(self.l1(x))

        x = self.nn.functional.leaky_relu(self.d1(code))
        x = self.nn.functional.relu(self.d2(x))
        x = self.nn.functional.relu(self.d3(x))
        x = x.view(x.shape[0], 64, 8, 8)

        x = torch.nn.functional.leaky_relu(self.convd1(self.tmp(x)))
        x = self.nn.functional.sigmoid(self.convd2(self.tmp2(x)))

        return x, code


import torchvision

net = MyAutoencoder()
net = net.cuda()

print("load data")
trainset = torchvision.datasets.CIFAR10(
    root="build",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
testset = torchvision.datasets.CIFAR10(
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
    print("train accuracy", L1 / nb)

print("eval model")
L1, nb = compute_accuracy(testloader, net)
print("test accuracy", L1 / nb)
