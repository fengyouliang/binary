from torch import optim

from register import Registry

register_opt = Registry('optimizer')


@register_opt.register()
def sgd():
    opt = optim.SGD
    return opt


@register_opt.register()
def adam():
    opt = optim.Adam
    return opt


@register_opt.register()
def rmsprop():
    opt = optim.RMSprop
    return opt


@register_opt.register()
def adamw():
    opt = optim.AdamW
    return opt
