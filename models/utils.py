from typing import Callable

import torch.nn as nn
from fastai.torch_core import requires_grad, bn_types, SplitFuncOrIdxList, split_model, apply_init
from fastai.vision.learner import create_cnn_model, cnn_config


class CNNPretrainedModel(nn.Module):
    """
    Customized from fastai learner
    """

    def __init__(self, base_arch, no_classes, dropout=0.5, init=nn.init.kaiming_normal_):
        super(CNNPretrainedModel, self).__init__()

        self.model = create_cnn_model(base_arch, no_classes, ps=dropout)
        self.meta = cnn_config(base_arch)
        self.split(self.meta['split'])
        self.freeze()

        apply_init(self.model[1], init)

    def split(self, split_on:SplitFuncOrIdxList)->None:
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on(self.model)
        self.layer_groups = split_model(self.model, split_on)
        return self

    def freeze_to(self, n:int)->None:
        "Freeze layers up to layer group `n`."
        for g in self.layer_groups[:n]:
            for l in g:
                if not isinstance(l, bn_types): requires_grad(l, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)

    def freeze(self)->None:
        "Freeze up to last layer group."
        assert(len(self.layer_groups)>1)
        self.freeze_to(-1)

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def forward(self, x):
        return self.model.forward(x)

