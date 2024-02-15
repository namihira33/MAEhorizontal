import torchvision.models as models
import torch.nn as nn
import torch
import config
import timm
from torchvision import transforms

device = torch.device('cuda:0')

class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=1)
        self.net.features[0] = nn.Conv2d(1,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Vgg16_bn(nn.Module):
    def __init__(self,n_per_unit):
        super().__init__()
        self.net = models.vgg16_bn()
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=config.n_class)
        self.net.features[0] = nn.Conv2d(n_per_unit,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Vgg19_bn(nn.Module):
    def __init__(self,n_per_unit):
        super().__init__()
        self.net = models.vgg19_bn()
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=config.n_class)
        self.net.features[0] = nn.Conv2d(n_per_unit,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet34()
        self.net.conv1 = nn.Conv2d(1,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=512,out_features=config.n_class)

    def forward(self,x):
        return self.net(x)

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18()
        self.net.conv1 = nn.Conv2d(1,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=512,out_features=config.n_class)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        #x = self.softmax(x)
        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = 'vit_base_patch16_224'
        self.net = timm.create_model(model_name,num_classes=config.n_class)
        #self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = self.net(x)
        #x = self.softmax(x)
        return x

class ViT_1kF(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = 'vit_base_patch16_224'
        self.net = timm.create_model(model_name,num_classes=config.n_class,pretrained=True)

        update_param_names = ['head.weight','head.bias']

        for name,param in self.net.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
            else:
                param.requires_grad = False
        #self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = self.net(x)
        #x = self.softmax(x)
        return x

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self,global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]

        #パッチをエンべディング
        x = self.patch_embed(x)

        #クラストークンを追加・ポジション円べディングも追加
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)


        #トランスフォーマーブロックを適用
        for blk in self.blocks:
            x = blk(x)

        #normalization > outcomeにする過程で前結合そうを通すかどうか？
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome



def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=config.n_class,
        in_chans=config.n_per_unit,
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=config.n_class,
        in_chans=config.n_per_unit,
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=config.n_class,
        in_chans=config.n_per_unit,
        **kwargs
    )
    return model



class ViT_21kF(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = 'vit_base_patch16_224_in21k'
        self.net = timm.create_model(model_name,num_classes=config.n_class,pretrained=True)

        update_param_names = ['head.weight','head.bias']

        for name,param in self.net.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
            else:
                param.requires_grad = False
        #self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = self.net(x)
        #x = self.softmax(x)
        return x


def make_model(name,n_per_unit,encoder=None):
    if name == 'Vgg16':
        net = Vgg16()
    elif name == 'Vgg16_bn':
        net = Vgg16_bn(n_per_unit)
    elif name == 'Vgg19':
        net = Vgg19()
    elif name == 'Vgg19_bn':
        net = Vgg19_bn(n_per_unit)
    elif name == 'ResNet18':
        net = Resnet18().to(device)
    elif name == 'ResNet34':
        net = Resnet34().to(device)
    elif name == 'ViT':
        net = ViT().to(device)
    elif name == 'ViT_1k':
        net = ViT_1kF().to(device)
    elif name == 'ViT_21k':
        net = ViT_21kF().to(device)
    elif name == 'Swin':
        model_name = 'swin_base_patch4_window7_224_in22k'
        net = timm.create_model(model_name,pretrained=True,num_classes=config.n_class).to(device)
    elif name == 'MAE_ViT':
       net = vit_base_patch16()

    return net