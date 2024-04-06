import torch


def duplicate_zeros_like(source, device):
    target = {
        name: torch.zeros_like(source[name]).to(device)
        for name in source
    }
    return target


def duplicate(source, device):
    target = {
        name: torch.zeros_like(source[name]).to(device)
        for name in source
    }
    assign(target, source)
    return target


def assign(target, source):
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


def sub(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def sub_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone(
        ) - subtrahend[name].data.clone()


def mean(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack(
            [source[name].data for source in sources]),
                                       dim=0).clone()


def weighted_mean(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack(
            [m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()
