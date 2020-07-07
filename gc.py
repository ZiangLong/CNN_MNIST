import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

def _accuracy(true, pred, top_k=(1,)):

    max_k = max(top_k)
    batch_size = true.size(0)

    _, pred = pred.topk(max_k, 1)
    pred = pred.t()
    correct = pred.eq(true.view(1, -1).expand_as(pred))

    result = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.div_(batch_size).item())

    return result

def _evaluate(model, loss, val_iterator, n_validation_batches):

    loss_value = 0.0
    accuracy = 0.0
    total_samples = 0

    for j, (x_batch, y_batch) in enumerate(val_iterator):

        x_batch = Variable(x_batch.cuda(), volatile=True)
        y_batch = Variable(y_batch.cuda(async=True), volatile=True)
        n_batch_samples = y_batch.size()[0]
        logits = model(x_batch)

        # compute logloss
        batch_loss = loss(logits, y_batch).item()

        # compute accuracies
        pred = F.softmax(logits)
        batch_accuracy = _accuracy(y_batch, pred, top_k=(1,))[0]

        loss_value += batch_loss*n_batch_samples
        accuracy += batch_accuracy*n_batch_samples
        total_samples += n_batch_samples

        if j >= n_validation_batches:
            break

    return loss_value/total_samples, accuracy/total_samples


## load mnist dataset
from matplotlib import pyplot as plt
import numpy as np
use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 1000
test_batch_size = 1000

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=test_batch_size,
                shuffle=False)

## network

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        #qrelu = QReLU.apply
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## training
model = LeNet()

if use_cuda:
    model = model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = optim.SGD(model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()

mem = []
acc = []
test_acc = []

for epoch in range(20):
    # trainning
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        mem.append(loss.item())
        acc.append(_accuracy(target, F.softmax(out)))
    _, a = _evaluate(model,criterion,test_loader,test_batch_size)
    test_acc.append(a)

plt.rc('axes', linewidth=2)
fontsize = 14
plt.plot(mem)
plt.xlabel('Iterations',fontsize=fontsize,fontweight='bold')
plt.ylabel('loss',fontsize=fontsize,fontweight='bold')


ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
plt.savefig('./loss.png')