Forked from https://github.com/ikostrikov/pytorch-meta-optimizer :
PyTorch implementation of [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474).

Our idea is to apply similar mechanism to learn the optimizer online, not on a series of independent runs as in the original paper.
For now, we've experimented with a very simple variant of an online meta-optimizer, that only tunes its learning rate.
