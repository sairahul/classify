from typing import Callable

import torch.nn as nn
from fastai.callbacks.hooks import num_features_model
from fastai.torch_core import requires_grad, bn_types, SplitFuncOrIdxList, split_model, apply_init
from fastai.vision.learner import create_cnn_model, cnn_config, create_body, create_head


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


class CNNMultiTaskModel(nn.Module):
    """
    Customized from fastai learner
    """

    def __init__(self, base_arch, no_diseases, dropout=0.5, init=nn.init.kaiming_normal_):
        super(CNNPretrainedModel, self).__init__()

        self.body = create_body(base_arch, pretrained=True)
        nf = num_features_model(nn.Sequential(*self.body.children())) * 2

        self.disease_head = create_head(nf, no_diseases, ps=0.5, concat_pool=True, bn_final=False)
        #self.age_head = create_head(nf, 1, ps=0.5, concat_pool=True, bn_final=False)
        self.gender_head = create_head(nf, 2, ps=0.5, concat_pool=True, bn_final=False)
        #self.projection_head = create_head(nf, 3, ps=0.5, concat_pool=True, bn_final=False)

        self.disease_model = nn.Sequential(self.body, self.disease_head)

        self.meta = cnn_config(base_arch)
        self.split(self.meta['split'])
        self.freeze()

        apply_init(self.disease_head, init)
        #apply_init(self.age_head, init)
        apply_init(self.gender_head, init)
        #apply_init(self.projection_head, init)

    def split(self, split_on:SplitFuncOrIdxList)->None:
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on(self.disease_model)
        self.layer_groups = split_model(self.disease_model, split_on)
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
        bout = self.body.forward(x)
        disease_out = self.disease_head.forward(bout)
        gender_out = self.gender_head.forward(bout)
        return disease_out, gender_out

