import torch
from torch import optim
from optimizer.optimizer import create_Optimizer
from optimizer.scheduler import create_Scheduler
from optimizer.layer_optimizer import SeperateLayerParams

def get_optim_and_scheduler(model, network, epochs, lr, train_all=True, nesterov=False):
    if train_all:
        params = model.parameters()
    else:
        params = model.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    print("Step size: %d" % step_size)
    return optimizer, scheduler


def get_optim_and_scheduler_style(style_net, epochs, lr, nesterov=False, step_radio=0.8):
    optimizer = optim.SGD(style_net, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * step_radio)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d for style net" % step_size)
    return optimizer, scheduler


def get_optim_and_scheduler_layer_joint(style_net, epochs, lr, train_all=None, nesterov=False):
    optimizer = optim.SGD(style_net, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * 1.)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d for style net" % step_size)
    return optimizer, scheduler


def get_model_lr(name, model, fc_weight=1.0):
    if 'resnet' in name:
        return [
            (model.conv1, 1.0),  # 0
            (model.bn1, 1.0),  # 1
            (model.layer1, 1.0),  # 2

            (model.domain_discriminators[1], 1.0),  # 3
            (model.domain_attention[1],1.0),
            (model.wl_domain_discriminators[1], 1.0),
            (model.wl_spatial_attention[1], 1.0),
            # (model.concat_recover[1],1.0),


            (model.layer2, 1.0),  # 4
            (model.domain_discriminators[2], 1.0),  # 5
            (model.domain_attention[2],1.0),
            (model.wl_domain_discriminators[2], 1.0),
            (model.wl_spatial_attention[2], 1.0),
            # (model.concat_recover[2],1.0),

            (model.layer3, 1.0),  # 6
            (model.domain_discriminators[3], 1.0),  # 7
            (model.domain_attention[3],1.0),
            (model.wl_domain_discriminators[3], 1.0),
            (model.wl_spatial_attention[3], 1.0),
            # (model.concat_recover[3],1.0),
            #
            (model.layer4, 1.0),  # 8
            (model.domain_discriminators[4], 1.0),  # 9
            (model.domain_attention[4],1.0),
            (model.wl_domain_discriminators[4], 1.0),
            (model.wl_spatial_attention[4], 1.0),
            # (model.concat_recover[4],1.0),
            # (model.classifier, 1.0 * fc_weight)   # 10
            (model.my_mlp, 1.0)  # 10

        ]
    elif name == 'alexnet':
        return [
            (model.layer0, 1.0),  # 0
            (model.layer1, 1.0),  # 1
            (model.layer2, 1.0),  # 2
            (model.feature_layers, 1.0),  # 3
            (model.fc, 1.0 * fc_weight),  # 4
        ]
    else:
        raise NotImplementedError


def get_optimizer(model, init_lr, momentum=.9, weight_decay=.0005, nesterov=False):
    # optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay,
    #                       nesterov=nesterov)
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay,
                          nesterov=nesterov)
    return optimizer

# def get_optim_and_scheduler_scatter(model, network, epochs, lr,args, momentum=.9, weight_decay=.0005, nesterov=False, step_radio=0.8):  ## stablenet
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     optimizers = [  optimizer]
#
#     schedulers = [ ]
#     return optimizers, schedulers


# def get_optim_and_scheduler_scatter(model, network, epochs, lr,args, momentum=.9, weight_decay=.0005, nesterov=False, step_radio=0.8):  ## domain
#     model_lr = get_model_lr(name=network, model=model, fc_weight=1.0)
#     optimizers = [get_optimizer(model_part, lr * alpha, momentum, weight_decay, nesterov)
#                   for model_part, alpha in model_lr]
#     step_size = int(epochs * step_radio)
#     schedulers = [optim.lr_scheduler.StepLR(opt, step_size=step_size) for opt in optimizers]
#     return optimizers, schedulers

# def get_optim_and_scheduler_scatter(model, network, epochs, lr, args,momentum=.9, weight_decay=.0005, nesterov=False, step_radio=0.8): ## swin-transform
#     # model_lr = get_model_lr(name=network, model=model, fc_weight=1.0)
#     # optimizers = [get_optimizer(model_part, lr * alpha, momentum, weight_decay, nesterov)
#     #               for model_part, alpha in model_lr]
#     # step_size = int(epochs * step_radio)
#     # schedulers = [optim.lr_scheduler.StepLR(opt, step_size=step_size) for opt in optimizers]
#
#     hyp_cfg = {'epochs': epochs,
#             'lr0': lr,
#             'lrf_ratio': None,
#             'momentum': 0.937,
#             'weight_decay': 0.0005,
#             'warmup_momentum': 0.8,
#             'warm_ep': 3,
#             'loss': {'ce': True, 'bce': [False, 0.5, False]},
#             'label_smooth': 0.1,
#             'strategy': {'prog_learn': False, 'mixup': [0.01, [0, 70]], 'focal': [False, 0.25, 1.5],
#                          'ohem': [False, 8, 0.3, 255]},
#             'optimizer': ['sgd', False],
#             'scheduler': 'cosine_with_warm'}
#     params = SeperateLayerParams( model)
#
#
#     optimizer = create_Optimizer(optimizer= hyp_cfg['optimizer'][0], lr= hyp_cfg['lr0'],
#                                       weight_decay= hyp_cfg['weight_decay'],
#                                       momentum= hyp_cfg['warmup_momentum'],
#                                       params=params.create_ParamSequence(layer_wise= hyp_cfg['optimizer'][1],
#                                                                          lr= hyp_cfg['lr0']))
#     # scheduler
#     scheduler = create_Scheduler(scheduler= hyp_cfg['scheduler'], optimizer= optimizer,
#                                       warm_ep= hyp_cfg['warm_ep'], epochs= hyp_cfg['epochs'],
#                                       lr0= hyp_cfg['lr0'], lrf_ratio= hyp_cfg['lrf_ratio'])
#
#
#     optimizers = [optimizer]
#     schedulers = [scheduler]
#     return optimizers, schedulers
#
def get_optim_and_scheduler_scatter(model, network, epochs, lr, args, momentum=.9, weight_decay=.0005,
                                    nesterov=False, step_radio=0.8):  ## domain

    hyp_cfg = {'epochs': 200,
               'lr0': lr,
               'lrf_ratio': None,
               'momentum': 0.937,
               'weight_decay': 0.0005,
               'warmup_momentum': 0.8,
               'warm_ep': 3,
               'loss': {'ce': True, 'bce': [False, 0.5, False]},
               'label_smooth': 0.1,
               'strategy': {'prog_learn': False, 'mixup': [0.01, [0, 70]], 'focal': [False, 0.25, 1.5],
                            'ohem': [False, 8, 0.3, 255]},
               'optimizer': ['sgd', False],
               'scheduler': 'cosine_with_warm'}
    model_lr = get_model_lr(name=network, model=model, fc_weight=1.0)


    optimizers = [create_Optimizer(params=SeperateLayerParams( model_part).create_ParamSequence(layer_wise= hyp_cfg['optimizer'][1], lr= hyp_cfg['lr0'])   ,
                                   optimizer=hyp_cfg['optimizer'][0], lr=hyp_cfg['lr0'],
                                   weight_decay=hyp_cfg['weight_decay'],
                                   momentum=hyp_cfg['warmup_momentum']
                                   )
                  for model_part, alpha in model_lr]
    schedulers = [create_Scheduler(scheduler= hyp_cfg['scheduler'], optimizer= opt,
                                      warm_ep= hyp_cfg['warm_ep'], epochs= hyp_cfg['epochs'],
                                      lr0= hyp_cfg['lr0'], lrf_ratio= hyp_cfg['lrf_ratio']) for opt in optimizers]
    return optimizers, schedulers
