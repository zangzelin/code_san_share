import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
# from apex import amp, optimizers
from data_loader.get_loader import get_loader, get_loader_label, get_loader_ad, get_loader_amlp
from .utils import get_model_mme
from models.basenet import ResClassifier_MME, ResClassifier_MME_L2
from models.mlp import MLP, MLP_NOBN

def get_dataloaders(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    evaluation_data = kwargs["evaluation_data"]
    conf = kwargs["conf"]
    val_data = None
    if "val" in kwargs:
        val = kwargs["val"]
        if val:
            val_data = kwargs["val_data"]
    else:
        val = False

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader(source_data, target_data, evaluation_data,
                      data_transforms,
                      batch_size=conf.data.dataloader.batch_size,
                      return_id=True,
                      balanced=conf.data.dataloader.class_balance,
                      val=val, val_data=val_data)

def get_dataloaders_mlp(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    evaluation_data = kwargs["evaluation_data"]
    conf = kwargs["conf"]
    val_data = None
    if "val" in kwargs:
        val = kwargs["val"]
        if val:
            val_data = kwargs["val_data"]
    else:
        val = False

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader_amlp(source_data, target_data, evaluation_data,
                      data_transforms,
                      batch_size=conf.data.dataloader.batch_size,
                      return_id=True,
                      balanced=conf.data.dataloader.class_balance,
                      val=val, 
                      val_data=val_data, 
                      data_aug_crop=kwargs["data_aug_crop"],
                      aug_type=kwargs["aug_type"],
                      )


def get_dataloaders_ad(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    evaluation_data = kwargs["evaluation_data"]
    conf = kwargs["conf"]
    val_data = None
    if "val" in kwargs:
        val = kwargs["val"]
        if val:
            val_data = kwargs["val_data"]
    else:
        val = False

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader_ad(source_data, target_data, evaluation_data,
                      data_transforms,
                      batch_size=conf.data.dataloader.batch_size,
                      return_id=True,
                      balanced=conf.data.dataloader.class_balance,
                      val=val, val_data=val_data)


def get_dataloaders_label(source_data, target_data, target_data_label, evaluation_data, conf):

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        evaluation_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader_label(source_data, target_data, target_data_label,
                            evaluation_data, data_transforms,
                            batch_size=conf.data.dataloader.batch_size,
                            return_id=True,
                            balanced=conf.data.dataloader.class_balance)


class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.
    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def get_parameters(self):
        return [{"params": self.parameters(), "lr": 1.}]


def get_models_ad(kwargs):
    net = kwargs["network"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
    G, dim = get_model_mme(net, num_class=num_class)

    C2 = ResClassifier_MME_L2(num_classes=2 * num_class,
                           norm=False, input_size=dim)
    C1 = ResClassifier_MME(num_classes=num_class,
                           norm=False, input_size=dim)

    mlp = MLP(input_dim=2048, hidden_dim=4096, output_dim=256)

    nn_da = DomainDiscriminator(in_feature=2048, hidden_size=256)

    device = torch.device("cuda")
    G.to(device)
    C1.to(device)
    C2.to(device)
    mlp.to(device)
    nn_da.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
    opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                      weight_decay=0.0005, nesterov=True)
    opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                       momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                       nesterov=True)
    opt_mlp = optim.SGD(list(mlp.parameters()), lr=1.0,
                    momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                    nesterov=True)
    opt_nn_da = optim.SGD(list(nn_da.parameters()), lr=1.0,
                    momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                    nesterov=True)
    # [G, C1, C2, mlp], [opt_g, opt_c, opt_mlp] = amp.initialize([G, C1, C2, mlp],
    #                                               [opt_g, opt_c, opt_mlp],
    #                                               opt_level="O1")
    G = nn.DataParallel(G)
    C1 = nn.DataParallel(C1)
    C2 = nn.DataParallel(C2)
    mlp = nn.DataParallel(mlp)
    nn_da = nn.DataParallel(nn_da)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    param_lr_m = []
    for param_group in opt_mlp.param_groups:
        param_lr_m.append(param_group["lr"])
    param_lr_opt_da = []
    for param_group in opt_nn_da.param_groups:
        param_lr_opt_da.append(param_group["lr"])

    return G, C1, C2, mlp, nn_da, opt_g, opt_c, opt_mlp, opt_nn_da, param_lr_g, param_lr_c, param_lr_m, param_lr_opt_da

def get_models(kwargs):
    net = kwargs["network"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
    G, dim = get_model_mme(net, num_class=num_class)

    C2 = ResClassifier_MME_L2(num_classes=2 * num_class,
                           norm=False, input_size=dim)
    # C2 = ResClassifier_MME(num_classes=2 * num_class,
    #                        norm=False, input_size=dim, temp=1)
    
    C1 = ResClassifier_MME(num_classes=num_class,
                           norm=False, input_size=dim)

    mlp = MLP(input_dim=2048, hidden_dim=4096, output_dim=256)

    device = torch.device("cuda")
    G.to(device)
    C1.to(device)
    C2.to(device)
    mlp.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
    
    print('kwargs["optim_tool"] =', kwargs["optim_tool"])
    if kwargs["optim_tool"] == 'sgd':
        opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                        weight_decay=0.0005, nesterov=True)
        opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                        momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                        nesterov=True)
        opt_mlp = optim.SGD(list(mlp.parameters()), lr=1.0,
                        momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                        nesterov=True)
    if kwargs["optim_tool"] == 'adamw':
        opt_g = optim.AdamW(
            G.parameters(),
            lr=0.001,
            weight_decay=0.0000005,
            )
        opt_c = optim.AdamW(
            list(C1.parameters()) + list(C2.parameters()),
            lr=0.001,
            # momentum=conf.train.sgd_momentum, 
            weight_decay=0.0005,
            # nesterov=True
            )
        opt_mlp = optim.AdamW(
            list(mlp.parameters()), 
            lr=0.001,
            # momentum=conf.train.sgd_momentum,
            weight_decay=0.0005,
            # nesterov=True
            )

    # [G, C1, C2, mlp], [opt_g, opt_c, opt_mlp] = amp.initialize([G, C1, C2, mlp],
    #                                               [opt_g, opt_c, opt_mlp],
    #                                               opt_level="O1")
    G = nn.DataParallel(G)
    C1 = nn.DataParallel(C1)
    C2 = nn.DataParallel(C2)
    mlp = nn.DataParallel(mlp)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    param_lr_m = []
    for param_group in opt_mlp.param_groups:
        param_lr_m.append(param_group["lr"])

    return G, C1, C2, mlp, opt_g, opt_c, opt_mlp, param_lr_g, param_lr_c, param_lr_m



def get_models_amlp(kwargs):
    net = kwargs["network"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
    G, dim = get_model_mme(net, num_class=num_class)

    C2 = ResClassifier_MME_L2(input_size=2048, num_classes=2 * num_class,
                           norm=False)
    # C2 = ResClassifier_MME(num_classes=2 * num_class,
    #                        norm=False, input_size=dim, temp=1)
    
    C1 = ResClassifier_MME(input_size=2048, num_classes=num_class,
                           norm=False)

    mlp = MLP(input_dim=2048, hidden_dim=4096, output_dim=2048)
    # mlp = MLP_NOBN(input_dim=2048, hidden_dim=4096, output_dim=2048)
    

    device = torch.device("cuda")
    G.to(device)
    C1.to(device)
    C2.to(device)
    mlp.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
    
    print('kwargs["optim_tool"] =', kwargs["optim_tool"])
    if kwargs["optim_tool"] == 'sgd':
        opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                        weight_decay=0.0005, nesterov=True)
        opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                        momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                        nesterov=True)
        opt_mlp = optim.SGD(
            list(mlp.parameters()), 
            lr=0.01,
            # momentum=conf.train.sgd_momentum,
            weight_decay=0.00005,
            # nesterov=True
            )
    if kwargs["optim_tool"] == 'adamw':
        opt_g = optim.AdamW(
            G.parameters(),
            lr=0.001,
            weight_decay=0.0000005,
            )
        opt_c = optim.AdamW(
            list(C1.parameters()) + list(C2.parameters()),
            lr=0.001,
            # momentum=conf.train.sgd_momentum, 
            weight_decay=0.0005,
            # nesterov=True
            )
        opt_mlp = optim.AdamW(
            list(mlp.parameters()), 
            lr=0.001,
            # momentum=conf.train.sgd_momentum,
            weight_decay=0.0005,
            # nesterov=True
            )

    # [G, C1, C2, mlp], [opt_g, opt_c, opt_mlp] = amp.initialize([G, C1, C2, mlp],
    #                                               [opt_g, opt_c, opt_mlp],
    #                                               opt_level="O1")
    G = nn.DataParallel(G)
    C1 = nn.DataParallel(C1)
    C2 = nn.DataParallel(C2)
    mlp = nn.DataParallel(mlp)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    param_lr_m = []
    for param_group in opt_mlp.param_groups:
        param_lr_m.append(param_group["lr"])

    return G, C1, C2, mlp, opt_g, opt_c, opt_mlp, param_lr_g, param_lr_c, param_lr_m




def get_models_amlp_oda(kwargs):
    net = kwargs["network"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
    G, dim = get_model_mme(net, num_class=num_class)

    C2 = ResClassifier_MME_L2(input_size=2048, num_classes=2 * num_class,
                           norm=False)
    # C2 = ResClassifier_MME(num_classes=2 * num_class,
    #                        norm=False, input_size=dim, temp=1)
    
    C1 = ResClassifier_MME(input_size=2048, num_classes=num_class,
                           norm=False)

    mlp = MLP(input_dim=2048, hidden_dim=4096, output_dim=2048)
    # mlp = MLP_NOBN(input_dim=2048, hidden_dim=4096, output_dim=2048)
    

    device = torch.device("cuda")
    G.to(device)
    C1.to(device)
    C2.to(device)
    mlp.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
    
    print('kwargs["optim_tool"] =', kwargs["optim_tool"])
    if kwargs["optim_tool"] == 'sgd':
        opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                        weight_decay=0.0005, nesterov=True)
        opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                        momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                        nesterov=True)
        opt_mlp = optim.SGD(
            list(mlp.parameters()), 
            lr=0.01,
            # momentum=conf.train.sgd_momentum,
            weight_decay=kwargs["mlp_weight_decay"],
            # nesterov=True
            )
    if kwargs["optim_tool"] == 'adamw':
        opt_g = optim.AdamW(
            G.parameters(),
            lr=0.001,
            weight_decay=0.0000005,
            )
        opt_c = optim.AdamW(
            list(C1.parameters()) + list(C2.parameters()),
            lr=0.001,
            # momentum=conf.train.sgd_momentum, 
            weight_decay=0.0005,
            # nesterov=True
            )
        opt_mlp = optim.AdamW(
            list(mlp.parameters()), 
            lr=0.001,
            # momentum=conf.train.sgd_momentum,
            weight_decay=0.0005,
            # nesterov=True
            )

    # [G, C1, C2, mlp], [opt_g, opt_c, opt_mlp] = amp.initialize([G, C1, C2, mlp],
    #                                               [opt_g, opt_c, opt_mlp],
    #                                               opt_level="O1")
    G = nn.DataParallel(G)
    C1 = nn.DataParallel(C1)
    C2 = nn.DataParallel(C2)
    mlp = nn.DataParallel(mlp)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    param_lr_m = []
    for param_group in opt_mlp.param_groups:
        param_lr_m.append(param_group["lr"])

    return G, C1, C2, mlp, opt_g, opt_c, opt_mlp, param_lr_g, param_lr_c, param_lr_m


def get_models_amlp_v2(kwargs):
    net = kwargs["network"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
    G, dim = get_model_mme(net, num_class=num_class)

    C2 = ResClassifier_MME_L2(input_size=2048, num_classes=2 * num_class,
                           norm=False)
    # C2 = ResClassifier_MME(num_classes=2 * num_class,
    #                        norm=False, input_size=dim, temp=1)
    
    C1 = ResClassifier_MME(input_size=2048, num_classes=num_class,
                           norm=False)

    # mlp = MLP(input_dim=2048, hidden_dim=4096, output_dim=2048)
    mlp = MLP_NOBN(input_dim=2048, hidden_dim=4096, output_dim=2048)
    

    device = torch.device("cuda")
    G.to(device)
    C1.to(device)
    C2.to(device)
    mlp.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
    
    print('kwargs["optim_tool"] =', kwargs["optim_tool"])
    if kwargs["optim_tool"] == 'sgd':
        opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                        weight_decay=0.0005, nesterov=True)
        opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                        momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                        nesterov=True)
        opt_mlp = optim.SGD(
            list(mlp.parameters()), 
            lr=0.01,
            # momentum=conf.train.sgd_momentum,
            weight_decay=0.00005,
            # nesterov=True
            )
    if kwargs["optim_tool"] == 'adamw':
        opt_g = optim.AdamW(
            G.parameters(),
            lr=0.001,
            weight_decay=0.0000005,
            )
        opt_c = optim.AdamW(
            list(C1.parameters()) + list(C2.parameters()),
            lr=0.001,
            # momentum=conf.train.sgd_momentum, 
            weight_decay=0.0005,
            # nesterov=True
            )
        opt_mlp = optim.AdamW(
            list(mlp.parameters()), 
            lr=0.001,
            # momentum=conf.train.sgd_momentum,
            weight_decay=0.0005,
            # nesterov=True
            )

    # [G, C1, C2, mlp], [opt_g, opt_c, opt_mlp] = amp.initialize([G, C1, C2, mlp],
    #                                               [opt_g, opt_c, opt_mlp],
    #                                               opt_level="O1")
    G = nn.DataParallel(G)
    C1 = nn.DataParallel(C1)
    C2 = nn.DataParallel(C2)
    mlp = nn.DataParallel(mlp)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    param_lr_m = []
    for param_group in opt_mlp.param_groups:
        param_lr_m.append(param_group["lr"])

    return G, C1, C2, mlp, opt_g, opt_c, opt_mlp, param_lr_g, param_lr_c, param_lr_m



def get_models_half(kwargs):
    net = kwargs["network"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
    G, dim = get_model_mme(net, num_class=num_class)

    C2 = ResClassifier_MME_L2(num_classes=2 * num_class,
                           norm=False, input_size=dim)
    C1 = ResClassifier_MME(num_classes=num_class,
                           norm=False, input_size=dim)

    mlp = MLP(input_dim=2048, hidden_dim=4096, output_dim=256)

    device = torch.device("cuda")
    G.half()
    C1.half()
    C2.half()
    mlp.half()
    
    G.to(device)
    C1.to(device)
    C2.to(device)
    mlp.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
    
    print('kwargs["optim_tool"] =', kwargs["optim_tool"])
    if kwargs["optim_tool"] == 'sgd':
        opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                        weight_decay=0.0005, nesterov=True)
        opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                        momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                        nesterov=True)
        opt_mlp = optim.SGD(list(mlp.parameters()), lr=1.0,
                        momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                        nesterov=True)
    if kwargs["optim_tool"] == 'adamw':
        opt_g = optim.AdamW(params, 
                        weight_decay=0.0000005,
                        # momentum=conf.train.sgd_momentum,
                        # nesterov=True
                        )
        opt_c = optim.AdamW(list(C1.parameters()) + list(C2.parameters()),
                        lr=0.001,
                        weight_decay=0.0000005,
                        # momentum=conf.train.sgd_momentum,
                        # nesterov=True
                        )
        opt_mlp = optim.AdamW(list(mlp.parameters()),
                        lr=0.001,
                        weight_decay=0.0000005,
                        # momentum=conf.train.sgd_momentum,
                        # nesterov=True
                        )

    # [G, C1, C2, mlp], [opt_g, opt_c, opt_mlp] = amp.initialize([G, C1, C2, mlp],
    #                                               [opt_g, opt_c, opt_mlp],
    #                                               opt_level="O1")
    G = nn.DataParallel(G)
    C1 = nn.DataParallel(C1)
    C2 = nn.DataParallel(C2)
    mlp = nn.DataParallel(mlp)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    param_lr_m = []
    for param_group in opt_mlp.param_groups:
        param_lr_m.append(param_group["lr"])

    return G, C1, C2, mlp, opt_g, opt_c, opt_mlp, param_lr_g, param_lr_c, param_lr_m