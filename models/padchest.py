import os

from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning import Trainer
from albumentations.pytorch import ToTensor

from sklearn import metrics
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from fastai.vision.learner import create_cnn_model
from fastai.vision.models import resnet34
from fastai.layers import FlattenedLoss
import albumentations as A
import PIL
from utils import CNNPretrainedModel
import matplotlib.pyplot as plt


def plot_rocauc(fpr, tpr, roc_auc):
    figure = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return figure

def open_image_png16(fn):
    a = np.array(PIL.Image.open(fn), dtype=np.uint16)
    a = a.astype('float32')
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.repeat(a, 3, axis=2)
    a = a / 65535.0
    return a


class ImageListDataset(Dataset):
    def __init__(self, base_folder, image_list, mapping, transforms):
        self.base_folder = base_folder
        self.image_list = image_list
        self.mapping = mapping
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        lbl = self.mapping[self.image_list[idx]]
        return {
            'image': self.transforms(image=open_image_png16(self.base_folder/self.image_list[idx]))['image'],
            'label': lbl
        }

class ChestXRayDiagnosis(pl.LightningModule):

    def __init__(self, transforms, params):
        super(ChestXRayDiagnosis, self).__init__()
        # not the best model...
        # Model, no_classes,
        #self.model = create_cnn_model(resnet34, 2, pretrained=True, ps=0.5, concat_pool=True)

        self.params = params
        self.resnet34 = CNNPretrainedModel(resnet34, 2)

        #self.model10 = CNNPretrainedModel(resnet34, 2)
        #self.cnn_model = self.model10.get_cnn_model()

        self.path = Path("/media/disk4tb/datasets/medical/chest/PC")
        df_clean = pd.read_csv(self.path/'cleaned_padchest.csv')

        diagnosis = ['pneumonia', 'atypical pneumonia', 'tuberculosis', 'tuberculosis sequelae',
                 'lung metastasis', 'lymphangitis carcinomatosa', 'lepidic adenocarcinoma', 'pulmonary fibrosis',
                 'post radiotherapy changes', 'asbestosis signs', 'emphysema', 'COPD signs', 'heart insufficiency',
                 'respiratory distress', 'pulmonary hypertension', 'pulmonary artery hypertension', 'pulmonary venous hypertension',
                 'pulmonary edema', 'bone metastasis']

        self.transforms = transforms
        self.disease_labels = {}
        for i, row in df_clean.iterrows():
            lbls = set(eval(row.Labels))
            if lbls.intersection(diagnosis):
                self.disease_labels[row.ImageID] = np.array([1])
            elif len(lbls) == 1 and 'normal' in lbls:
                self.disease_labels[row.ImageID] = np.array([0])
            else:
                pass

        allimages = set(list(self.disease_labels.keys()))
        with open(self.path/"train_val_test.pkl", "rb") as fp:
            _, val_, _ = pickle.load(fp)

        val_ = set(val_)
        self.train_set = allimages - val_
        self.val_set = allimages.intersection(val_)
        self.classes = {
            0: 'normal',
            1: 'disease'
        }
        self.loss_func = FlattenedLoss(nn.CrossEntropyLoss)

    def create_lr_scheduler(self, each_step, optimizer):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, each_step['lr'], steps_per_epoch=self.steps_per_epoch, epochs=each_step['epochs'])
        return scheduler

    def forward(self, x):
        #return self.cnn_model.forward(x)
        return self.resnet34(x)

    def training_step(self, batch, batch_idx):
        if self.trainer.batch_idx == 0:
            if self.trainer.current_epoch == self.params['stages'][0]['epochs']:
                print('Starting first ....')
                model.freeze_to(-2)
                self.trainer.lr_schedulers[0]['scheduler'] = self.create_lr_scheduler(self.params['stages'][1], self.trainer.optimizers[0])
            elif self.trainer.current_epoch == self.params['stages'][0]['epochs'] + self.params['stages'][1]['epochs']:
                print('Starting second ....')
                model.freeze_to(0)
                self.trainer.lr_schedulers[0]['scheduler'] = self.create_lr_scheduler(self.params['stages'][2], self.trainer.optimizers[0])

        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)

        loss = self.loss_func(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)
        return {'val_loss': self.loss_func(y_hat, y), 'y_hat': y_hat, 'y': y}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        y_pred = F.softmax(torch.cat([x['y_hat'] for x in outputs]), dim=-1)
        y = torch.cat([x['y'] for x in outputs])

        fpr, tpr, threshold = metrics.roc_curve(y.cpu(), y_pred[:, 1].cpu())
        roc_auc = metrics.auc(fpr, tpr)

        tensorboard_logs = {
            'val_loss': avg_loss,
            #'fpr': fpr,
            #'tpr': tpr,
            'auc': roc_auc
        }

        fig_rocauc = plot_rocauc(fpr, tpr, roc_auc)
        self.logger.experiment.add_figure('ROC/AUC', fig_rocauc)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        train_dl = self.train_dataloader()
        self.steps_per_epoch = len(train_dl)

        schedulers = []
        current_epochs = 0
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        schedulers.append({
            "scheduler": self.create_lr_scheduler(self.params['stages'][0], optimizer),
            "interval" : "step"
            })
        return [optimizer], schedulers

    def train_dataloader(self):
        return DataLoader(ImageListDataset(self.path/'images-224/', list(self.train_set), self.disease_labels, transforms=self.transforms), batch_size=self.params['batch_size'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(ImageListDataset(self.path/'images-224/', list(self.val_set), self.disease_labels, transforms=self.transforms), batch_size=self.params['batch_size'], shuffle=True, num_workers=4)

    def freeze_to(self, n):
        self.resnet34.freeze_to(n)


if __name__=="__main__":

    tfms =  A.Compose([
            A.HorizontalFlip(),
            A.OneOf([
                A.RandomContrast(),
                A.RandomGamma(),
                A.RandomBrightness(),
                ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
            A.ShiftScaleRotate(),
            A.Normalize(max_pixel_value=1.0, mean=(0.521,0.521,0.521), std=(0.304,0.304,0.304)),
            ToTensor(),
        ])

    params = {
        "batch_size": 32,
        "stages": [
            {
                "epochs": 4,
                "lr": 0.001,
                "freeze_to": -1
            },
            {
                "epochs": 4,
                "lr": 0.0001,
                "freeze_to": -2
            },
            {
                "epochs": 8,
                "lr": 0.00001,
                "freeze_to": 0
            }
        ]
    }
    model = ChestXRayDiagnosis(tfms, params)

    # most basic trainer, uses good defaults
    #trainer = Trainer(max_epochs=16, gpus=1, callbacks=[TransferLearningTuner()], accumulate_grad_batches=2)
    trainer = Trainer(max_epochs=16, gpus=1)
    trainer.fit(model)

