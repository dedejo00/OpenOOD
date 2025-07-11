import torch.nn as nn
from openood.networks.de_resnet18_256x256 import De_ResNet18_256x256
from openood.networks.nca import BasicNCAModel


class ReactNet(nn.Module):
    def __init__(self, backbone):
        super(ReactNet, self).__init__()
        self.backbone = backbone

    def forward(self, x, return_feature=False, return_feature_list=False):
        try:
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            return self.backbone(x, return_feature)

    def forward_threshold(self, x, threshold):
        if isinstance(self.backbone, BasicNCAModel):
            self.backbone.threshold_activations_react = threshold
            logits = self.backbone.classify(x, 72, False)
            return logits
        else:
            _, feature = self.backbone(x, return_feature=True)
            feature = feature.clip(max=threshold)
            feature = feature.view(feature.size(0), -1)
            logits_cls = self.backbone.get_fc_layer()(feature)
            return logits_cls

    def get_fc(self):
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
