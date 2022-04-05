import torch


def compute_accuracy(batchprovider, net):
    with torch.no_grad():
        net.eval()
        net = net.cuda()
        nb, nbOK = 0, 0
        for x, y in batchprovider:
            x, y = x.cuda(), y.cuda()
            z = net(x)

            _, z = z.max(1)
            good = (y == z).float()
            nb += good.shape[0]
            nbOK += good.sum().cpu().numpy()
    return nbOK, nb


def training_epoch(batchprovider, net, lr):
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    meanloss = 0
    nb, nbOK = 0, 0
    for i, (x, y) in enumerate(batchprovider):
        x, y = x.cuda(), y.cuda()
        z = net(x)

        loss = criterion(z, y)

        with torch.no_grad():
            meanloss += loss.clone().cpu().numpy()
            _, z = z.max(1)
            good = (y == z).float()
            nb += good.shape[0]
            nbOK += good.sum().cpu().numpy()
            if i % 50 == 49:
                print("loss=", meanloss / 50)
                meanloss = 0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
        optimizer.step()

    return nbOK, nb


import torchvision

print("load model")
net = torchvision.models.vgg13(pretrained=True)
net.avgpool = torch.nn.Identity()
net.classifier = torch.nn.Linear(512, 10)
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
    testset, batch_size=256, shuffle=True, num_workers=2
)

print("finetune the model on the data")

for epoch in range(8):
    print("epoch", epoch)
    nbOK, nb = training_epoch(trainloader, net, 0.0001)
    print("train accuracy", nbOK / nb)

print("eval model")
nbok, nb = compute_accuracy(testloader, net)
print("test accuracy", nbok / nb)


print("show example")
x, y = next(iter(testloader))
with torch.no_grad():
    z = net(x.cuda())
    _, z = z.max(1)
z = z.cpu()

for i in range(x.shape[0]):
    if z[i] == y[i]:
        x[i, 1, :, :] = 1  # bon - on met l'image en vert
    else:
        x[i, 0, :, :] = 1  # pas bon - on met l'image en rouge

visu = torchvision.utils.make_grid(x, nrow=16)
torchvision.utils.save_image(visu, "visu.png")
