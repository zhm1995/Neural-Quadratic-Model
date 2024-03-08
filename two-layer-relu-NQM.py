import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import argparse
import math
import copy
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--width', default=1024, type=int,
                        help='Width of neural network')
    parser.add_argument('--input_dim', default=3072, type=int,
                        help='Dimension of the input data')
    parser.add_argument('--id', default=0, type=int,
                        help='Binary classes selected for training')
    parser.add_argument('--lr', default=3, type=float, help='Learning rate')
    parser.add_argument('--seed', default=7, type=int, help='Random seed')
    parser.add_argument('--train-size', default=256,
                        type=int, help='Train size')
    parser.add_argument('--test-size', default=1000,
                        type=int, help='Test size')
    parser.add_argument('--batch-size', default=1000,
                        type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--log-interval', default=10,
                        type=int, help='Logging interval')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    return parser.parse_args()

#Two layer neural network with ReLU activation function, using NTK parameterization
class FCN(nn.Module):
    def __init__(self, input_dim, width):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 1)
        nn.init.normal_(self.fc2.weight, 0, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = math.sqrt(1 / x.size(1)) * self.fc1(x)
        activation = F.relu(x)
        x = math.sqrt(1 / self.fc1.out_features) * self.fc2(activation)
        return x, activation


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_data(id, train_size, test_size, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform)

    # Filter and subsample datasets
    idx = [i for i, x in enumerate(trainset.targets) if x == id or x == id+2]
    trainset.targets = [trainset.targets[i]-id-1 for i in idx][0:train_size]
    trainset.data = [trainset.data[i] for i in idx][0:train_size]

    idx = [i for i, x in enumerate(testset.targets) if x == id or x == id+2]
    testset.targets = [testset.targets[i]-id-1 for i in idx][0:test_size]
    testset.data = [testset.data[i] for i in idx][0:test_size]

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_quad_model(model, data_loader, input_dim, width, device):

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(
                device).view(-1, 1).float()
            output, activation = model(data)
            f_0 = output
            dfdw = 1 / math.sqrt(width) * 1 / math.sqrt(input_dim) * \
                (torch.gt(activation, 0)) * model.fc2.weight
            dfdv = 1 / math.sqrt(width) * activation
            df2dvdw = 1 / math.sqrt(width) * 1 / \
                math.sqrt(input_dim) * (torch.gt(activation, 0))
        return f_0, dfdw, dfdv, df2dvdw


def train(epoch, w, v, optimizer, data_loader, input_dim, f_0, dfdw, dfdv, df2dvdw, w_0, v_0, device):
    train_loss = AverageMeter()
    correct = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device).view(-1, 1).float()
        x = data.view(-1, input_dim)
        output = f_0 + torch.sum((x @ (w - w_0).t()) * dfdw, dim=1, keepdim=True) + dfdv @ (
            v - v_0).t() + torch.sum(((x @ (w - w_0).t()) * df2dvdw) @ (v - v_0).t(), dim=1,
                                     keepdim=True)
        diff = (output - target).t()
        w_derivative = (x.t() * diff) @ dfdw + \
            (x.t() * diff) @ (df2dvdw * (v - v_0))
        v_derivative = dfdv.t() @ (output - target) + ((x @ (w - w_0).t()) * df2dvdw).t() @ (
            output - target)

        optimizer.zero_grad()
        w.grad = 2 * w_derivative.t() / len(data_loader.dataset)
        v.grad = 2 * v_derivative.t() / len(data_loader.dataset)
        pred = torch.sign(output)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = F.mse_loss(output, target)
        train_loss.update(loss.item(), len(data))
        optimizer.step()
    print(f'step: {epoch}, training_loss: {train_loss.avg}, training_accu: {correct / len(data_loader.dataset)}')


def test(epoch, w, v, optimizer, data_loader, input_dim, f_0, dfdw, dfdv, df2dvdw, w_0, v_0, device):
    test_loss = AverageMeter()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(
                device).view(-1, 1).float()
            x_t = data.view(-1, 3072)
            output_test = f_0 + torch.sum((x_t @ (w - w_0).t()) * dfdw, dim=1, keepdim=True) + dfdv @ (
                v - v_0).t() + torch.sum(((x_t @ (w - w_0).t()) * df2dvdw) @ (v - v_0).t(), dim=1,
                                         keepdim=True)
            loss = F.mse_loss(output_test, target)
            test_loss.update(loss.item(), len(data))
            pred = torch.sign(output_test)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(
        f'step: {epoch}, test_loss: {test_loss.avg}, test_accu: {correct / len(data_loader.dataset)}')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(
        args.id, args.train_size, args.test_size, args.batch_size)

    model = FCN(args.input_dim, args.width).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    w_0 = copy.deepcopy(model.fc1.weight.data)
    v_0 = copy.deepcopy(model.fc2.weight.data)
    w = model.fc1.weight
    v = model.fc2.weight
    f_0, dfdw, dfdv, df2dvdw = get_quad_model(
        model, train_loader, args.input_dim, args.width, device)
    f_t, dfdw_t, dfdv_t, df2dvdw_t = get_quad_model(
        model, test_loader, args.input_dim, args.width, device)

    for epoch in range(1, args.epochs + 1):
        # Training and testing loops
        train(epoch, w, v, optimizer, train_loader, args.input_dim,
              f_0, dfdw, dfdv, df2dvdw, w_0, v_0, device)
        test(epoch, w, v, optimizer, test_loader, args.input_dim,
             f_t, dfdw_t, dfdv_t, df2dvdw_t, w_0, v_0, device)

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    main()
