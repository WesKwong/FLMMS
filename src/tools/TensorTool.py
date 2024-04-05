import torch


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def add_(target, source1, source2):
    for name in target:
        target[name].data = source1[name].data.clone(
        ) + source2[name].data.clone()


def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone(
        ) - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack(
            [source[name].data for source in sources]),
                                       dim=0).clone()


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack(
            [m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()