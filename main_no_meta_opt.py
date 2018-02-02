import argparse
import operator
import sys
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_batch
from model import ModelDefault, ModelFullyConnected
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='number of BPTT steps (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=100, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=10000, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--use_sgd', action='store_true', default=False,
                    help='use plain sgd instead of Adam')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

assert args.optimizer_steps % args.truncated_bptt_step == 0

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

Model = ModelFullyConnected
kwargs = {'num_hidden': 2, 'hidden_size': 256, 'dropout_rate': 0.5}

def main():
    alpha = 0.999

    d = 1

    start_time = time()

    for epoch in range(args.max_epoch):
        decrease_in_loss = 0.0
        final_loss = 0.0
        train_iter = iter(train_loader)
        for i in range(args.updates_per_epoch):

            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # Sample a new model
            model = Model(**kwargs)
            if args.cuda:
                model.cuda()

            if args.use_sgd:
                optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            else:
                optimizer = optim.Adam(model.parameters(), lr=1e-5)

            # Compute initial loss of the model
            f_x = model(x)
            initial_loss = F.nll_loss(f_x, y)

            av_loss = 0.

            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                for j in range(args.truncated_bptt_step):
                    try:
                        x, y = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        x, y = next(train_iter)

                    if args.cuda:
                        x, y = x.cuda(), y.cuda()
                    x, y = Variable(x), Variable(y)

                    # First we need to compute the gradients of the model
                    f_x = model(x)
                    loss = F.nll_loss(f_x, y)
                    model.zero_grad()
                    loss.backward()

                    optimizer.step()

                    av_loss = alpha * av_loss + (1 - alpha) * loss.data
                    
                print('av_loss = {:.3f}'.format(av_loss[0]))
                if av_loss[0] < 0.1**d:
                    print(
                        'model reached loss < 1e-{} in {} steps ({:.1f}s)'.format(
                            d, k * args.truncated_bptt_step, time() - start_time
                        )
                    )
                    if d >= 3:
                        break
                    d += 1


            # Compute relative decrease in the loss function w.r.t initial
            # value
            decrease_in_loss += loss.data[0] / initial_loss.data[0]
            final_loss += loss.data[0]

        print(
            "Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(
                epoch, final_loss / args.updates_per_epoch, decrease_in_loss / args.updates_per_epoch
            )
        )

if __name__ == "__main__":
    main()
