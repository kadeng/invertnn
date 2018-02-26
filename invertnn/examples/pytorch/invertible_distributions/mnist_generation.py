
from __future__ import print_function
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
import torch.distributions as distributions
from sklearn import datasets

# In[19]:

import invertnn.pytorch.invertible_transforms as invt


def random_covariance(n):
    tmp = torch.rand((n,n)).sqrt().mul(2.0).sub(1.0).triu()
    k = torch.matmul(tmp, tmp.t())
    for i in range(n):
        k[i,i] = abs(k[i,i])
    return k

def create_mv_normal(dims=2):
    return distributions.MultivariateNormal(torch.rand((dims)),covariance_matrix=random_covariance(dims))

def simple_transform(tensor : torch.Tensor):
    tensor[:, 1] = torch.where(tensor[:, 0] > 0, tensor[:, 0], tensor[:, 0].pow(2) * 0.002)
    return tensor


class SimpleConvnet(nn.Module):

    def __init__(self, n, m, hidden=[50,50,20], final_activation=None):
        super().__init__()
        self.n = n
        self.m = m
        self.final_activation = final_activation
        self.net = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(n, 10, kernel_size=5, padding=2))
            ('a1', nn.LeakyReLU()),
            ('bn1', nn.BatchNorm2d(10)),
            ('c2', nn.Conv2d(10, 20, kernel_size=5, padding=2))
            ('a2', nn.LeakyReLU()),
            ('bn2', nn.BatchNorm2d(10)),
            ('c3', nn.Conv2d(20, 20, kernel_size=3, padding=1))
            ('a3', nn.LeakyReLU()),
            ('bn3', nn.BatchNorm2d(10)),
            ('c4', nn.Conv2d(20, m, kernel_size=3, padding=1))
            ('a4', nn.LeakyReLU())
        ]))

    def forward(self, input):
        active = self.net(input)
        if self.final_activation is not None:
            return self.final_activation(active)
        else:
            return active

def create_invertible_net():
    # We constrain the output range of s via Softsign, because it's result gets exponentiated
    invnn = invt.InvertibleSequential(OrderedDict([
        ('l1', invt.InvertibleCouplingLayer(D=2, d=1, s=SimpleConvnet(1,1, final_activation=nn.Softsign()), t=SimpleConvnet(2,1))),
        ('l2', invt.InvertibleCouplingLayer(D=2, d=1, s=SimpleConvnet(1,1, final_activation=nn.Softsign()), t=SimpleConvnet(2,1))),
        ('l3', invt.InvertibleCouplingLayer(D=2, d=1, s=SimpleConvnet(1,2, final_activation=nn.Softsign()), t=SimpleConvnet(1,2))),
        ('l4', invt.InvertibleCouplingLayer(D=2, d=1, s=SimpleConvnet(1,2, final_activation=nn.Softsign()), t=SimpleConvnet(1,2))),
        ('s5', InvertibleShuffle(permutation(np.arange(3)))),
        ('l5', invt.InvertibleCouplingLayer(D=2, d=1, s=SimpleConvnet(2,1, final_activation=nn.Softsign()), t=SimpleConvnet(2,1))),
    ]))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
