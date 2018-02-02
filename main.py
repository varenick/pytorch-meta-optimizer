import argparse
import operator
import sys
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_batch
from meta_optimizer import MetaModel, FastMetaOptimizer, MetaOptimizer, LearningRateOnlyMetaOptimizer
from model import ModelDefault, ModelFullyConnected, ModelConvolutional
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import copy_params

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=100, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=10000, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=10, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--fast_meta-opt', action='store_true', default=False,
                    help='switches to fast feedforward meta-optimizer')
parser.add_argument('--lr_only', action='store_true', default=False,
                    help='meta-optimize only learning rate')
parser.add_argument('--replay_trajectory', action='store_true', default=False,
                    help='replay last optimization trajectory once again after meta-optimizer parameters update')
parser.add_argument('--print_pause', type=int, default=1000, metavar='N',
                    help='optimizer step count between prints (default: 1000)')
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

#Model = ModelFullyConnected
#kwargs = {'num_hidden': 2, 'hidden_size': 256, 'dropout_rate': 0.5}
Model = ModelConvolutional
kwargs = {'num_hidden': 2, 'fc_hidden_size': 256, 'dropout_rate': 0.5, 'num_filters': 16, 'filter_size': 3}

def main():
    #torch.random.manual_seed(123)
    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.
    meta_model = Model(**kwargs)
    if args.cuda:
        meta_model.cuda()
    #for module in meta_model.modules():
    #    print(module._parameters)
    #    print(list(module.children()))

    if args.lr_only:
        meta_optimizer = LearningRateOnlyMetaOptimizer(MetaModel(meta_model), args.num_layers, args.hidden_size)
    elif args.fast_meta_opt:
        meta_optimizer = FastMetaOptimizer(MetaModel(meta_model), args.num_layers, args.hidden_size)
    else:
        meta_optimizer = MetaOptimizer(MetaModel(meta_model), args.num_layers, args.hidden_size)
    if args.cuda:
        meta_optimizer.cuda()

    optimizer = optim.Adam(meta_optimizer.parameters())

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
            
            if args.replay_trajectory:
                backup_model = Model(**kwargs)
                if args.cuda:
                    backup_model.cuda()

            # Compute initial loss of the model
            f_x = model(x)
            initial_loss = F.nll_loss(f_x, y)

            av_loss = 0.

            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                # Keep states for truncated BPTT
                meta_optimizer.reset_lstm(
                    keep_states=k>0, model=model, use_cuda=args.cuda
                )

                if args.replay_trajectory:
                    #meta_optimizer.backup_model_params()
                    copy_params(source=model, dest=backup_model)

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.cuda()
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

                    if not args.replay_trajectory:
                        av_loss = alpha * av_loss + (1 - alpha) * loss.data
                
                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the optimizer
                    meta_model = meta_optimizer.meta_update(model, loss.data)

                    # Compute a loss for a step the meta optimizer
                    f_x = meta_model(x)
                    loss = F.nll_loss(f_x, y)

                    loss_sum += (loss - Variable(prev_loss))

                    prev_loss = loss.data

                # Update the parameters of the meta optimizer
                meta_optimizer.zero_grad()
                #loss_sum.backward()
                loss.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                
                if args.replay_trajectory:
                    meta_optimizer.reset_lstm(
                        keep_states=k>0, model=backup_model, use_cuda=args.cuda
                    )
                    copy_params(source=backup_model, dest=model)

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

                        # Perfom a meta update using gradients from model
                        # and return the current meta model saved in the optimizer
                        meta_model = meta_optimizer.meta_update(model, loss.data)
                        
                        av_loss = alpha * av_loss + (1 - alpha) * loss.data

                if (k * args.truncated_bptt_step) % args.print_pause == 0:
                    if args.lr_only:
                        meta_optimizer.learning_rate.clamp(min=1e-8)
                        print('av_loss = {:.3f}; lr = {:.4f}'.format(av_loss[0], meta_optimizer.learning_rate.data[0]))
                    else:
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
