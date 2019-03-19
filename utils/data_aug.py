import sys
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils

crop_aug = gdata.vision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
brightness_aug = gdata.vision.transforms.RandomBrightness(0.5)
hue_aug = gdata.vision.transforms.RandomHue(0.5)
color_aug = gdata.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
lrflip_aug = gdata.vision.transforms.RandomFlipLeftRight()
tbflip_aug = gdata.vision.transforms.RandomFlipTopBottom()


