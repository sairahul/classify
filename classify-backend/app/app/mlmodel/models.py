
from fastai.vision import *
from app.mlmodel.losses import ContrastiveLoss


class SiameseNet(nn.Module):
    def __init__(self, arch=models.resnet34, lin_ftrs=[256, 128], emb_sz=128, ps=0.5, bn_final=False):
        super(SiameseNet, self).__init__()
        self.arch, self.emb_sz = arch, emb_sz
        self.lin_ftrs, self.ps, self.bn_final = lin_ftrs, ps, bn_final
        self.cnn = create_cnn_model(self.arch, emb_sz, cut=None, pretrained=True, lin_ftrs=lin_ftrs, ps=self.ps,
                                    bn_final=self.bn_final)

    def forward(self, x1, x2):
        output1 = self.cnn(x1)
        output2 = self.cnn(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.cnn(x)

    def __call__(self, x):
        return self.cnn(x)

def get_learner():
    model = SiameseNet()
    apply_init(model.cnn[1], nn.init.kaiming_normal_)
    loss_func = ContrastiveLoss()
    siam_learner = Learner(data, model, loss_func=loss_func, model_dir=Path(os.getcwd()),
                           layer_groups=[model.cnn[0], model.cnn[1]], opt_func=optim.SGD)
    return siam_learner

