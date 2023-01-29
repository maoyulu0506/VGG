from torch import nn
import torch
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

__all__ = ['vgg11','vgg11_bn','vgg13','vgg13_bn',
           'vgg16','vgg16_bn','vgg19','vgg19_bn']

models_url={
    'vgg11':"https://download.pytorch.org/models/vgg11-8a719046.pth",
    'vgg11_bn':"https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    'vgg13':"https://download.pytorch.org/models/vgg13-19584684.pth",
    'vgg13_bn':"https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    'vgg16':"https://download.pytorch.org/models/vgg16-397923af.pth",
    'vgg16_bn':"https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    'vgg19':"https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    'vgg19_bn':"https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"
}
cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class vgg(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.features = features
        
        #Initialize weights
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode = "fan_out",nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    
    def forward(self,x):
        feature_map=self.features(x)
        return feature_map
    
def _make_layer(cfg,batch_norm = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers +=[nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,stride=1,padding=1)
            if batch_norm:
                layers +=[conv2d,nn.BatchNorm2d(v),nn.ReLU()]
            else:
                layers +=[conv2d,nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)

def vgg11(pretrained = False,**kwargs):
    model = vgg(_make_layer(cfg["A"]))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(models_url['vgg11']),strict=False)
    return model

def vgg11_bn(pretrianed=False,**kwargs):
    model = vgg(_make_layer(cfg["A"],batch_norm=True))
    if pretrianed:
        model.load_state_dict(model_zoo.load_url(models_url['vgg11_bn']),strict=False)
    return model

def vgg13(pretrained=False,**kwargs):
    model = vgg(_make_layer(cfg["B"]))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(models_url['vgg13']),strict=False)
    return model

def vgg13_bn(pretrianed=False,**kwargs):
    model=vgg(_make_layer(cfg["B"],batch_norm=True))
    if pretrianed:
        model.load_state_dict(model_zoo.load_url(models_url['vgg13_bn']),strict=False)
    return model

def vgg16(pretrained=False,**kwargs):
    model=vgg(_make_layer(cfg["D"],batch_norm=True))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(models_url['vgg16']),strict=False)
    return model

def vgg16_bn(pretrained=False,**kwargs):
    model=vgg(_make_layer(cfg["D"],batch_norm=True))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(models_url['vgg16_bn']),strict=False)
    return model

def vgg19(pretrianed=False,**kwargs):
    model=vgg(_make_layer(cfg["E"]))
    if pretrianed:
        model.load_state_dict(model_zoo.load_url(models_url['vgg19']),strict=False)
    return model

def vgg19_bn(pretrained=False,**kwargs):
    model=vgg(_make_layer(cfg["E"],batch_norm=True))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(models_url['vgg19_bn']),strict=False)
    return model

if __name__ == '__main__':
    print(models.vgg11())
    model = vgg11(pretrained=True)
    print(model)
        
        
                

