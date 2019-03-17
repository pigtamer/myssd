import time, argparse
import mxnet as mx
import cv2 as cv
import matplotlib.pyplot as plt
from utils import utils, train, predata
from utils.utils import concat_preds, flatten_pred, hybrid_concat_preds
from mxnet import autograd, sym, init, nd, contrib, gluon, image
from mxnet.gluon import nn, trainer
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", dest="load",
                    help="int 0/1: load model to directly infer rather than training",
                    type=int, default=1)
parser.add_argument("-b", "--base", dest="base",
                    help="int 0/1: using additional base network",
                    type=int, default=0)
parser.add_argument("-s", "--imsize", dest="input_size",
                    help="int: input frame size",
                    type=int, default=256)
parser.add_argument("-m", "--model_path", dest="model_path",
                    help="str: the path to load and save model",
                    type=str, default="../params/myssd5.params")
parser.add_argument("-t", "--test_path", dest="test_path",
                    help="str: the path to your test img",
                    type=str, default="../data/uav/drone_video/Video_233.mp4")
args = parser.parse_args()


# TODO: read the calculation of following sizes!!
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


def genClsPredictor(num_cls, num_ach):
    return nn.Conv2D(channels=num_ach * (num_cls + 1), kernel_size=3, padding=1)


def genBBoxRegressor(num_ach):
    return nn.Conv2D(channels=num_ach * 4, kernel_size=3, padding=1)


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = nd.contrib.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


def hybrid_blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = sym.contrib.MultiBoxPrior(data=Y, sizes=size, ratios=ratio)
    # anchors = nd.contrib.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class MySSD(nn.HybridBlock):
    def __init__(self, num_cls, num_ach, **kwargs):
        super(MySSD, self).__init__(**kwargs)
        self.num_classes = num_cls
        self.BaseBlk = BaseNetwork(True)
        self.blk1 = nn.HybridSequential()
        self.blk1.add(nn.Conv2D(channels=1024, kernel_size=3, strides=1, padding=0),
                      nn.Conv2D(channels=1024, kernel_size=1, strides=1, padding=1),
                      nn.BatchNorm(in_channels=1024),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls1 = genClsPredictor(num_cls, num_ach)
        self.reg1 = genBBoxRegressor(num_ach)

        self.blk2 = nn.HybridSequential()
        self.blk2.add(nn.Conv2D(channels=256, kernel_size=1, strides=1, padding=0),
                      nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1),
                      nn.BatchNorm(in_channels=512),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls2 = genClsPredictor(num_cls, num_ach)
        self.reg2 = genBBoxRegressor(num_ach)

        self.blk3 = nn.HybridSequential()
        self.blk3.add(nn.Conv2D(channels=128, kernel_size=1, strides=1, padding=0),
                      nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
                      nn.BatchNorm(in_channels=256),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls3 = genClsPredictor(num_cls, num_ach)
        self.reg3 = genBBoxRegressor(num_ach)

        self.blk4 = nn.HybridSequential()
        self.blk4.add(nn.Conv2D(channels=128, kernel_size=1, strides=1, padding=0),
                      nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
                      nn.BatchNorm(in_channels=256),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls4 = genClsPredictor(num_cls, num_ach)
        self.reg4 = genBBoxRegressor(num_ach)

        self.blk5 = nn.HybridSequential()
        self.blk5.add(nn.Conv2D(channels=128, kernel_size=1, strides=1, padding=0),
                      nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
                      nn.BatchNorm(in_channels=256),
                      nn.Activation('relu'),
                      nn.MaxPool2D(2))

        self.cls5 = genClsPredictor(num_cls, num_ach)
        self.reg5 = genBBoxRegressor(num_ach)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.BaseBlk(x)

        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for k in range(5):
            if k == 0:
                im = x
            (x, anchors[k], cls_preds[k], bbox_preds[k]) = \
                hybrid_blk_forward(x, getattr(self, "blk%d" % (k + 1)), sizes[k], ratios[k],
                                   getattr(self, "cls%d" % (k + 1)), getattr(self, "reg%d" % (k + 1)))
            # print("layer[%d], fmap shape %s, anchor %s" % (k + 1, x.shape, anchors[k].shape))
        return (im, sym.concat(*anchors, dim=1),
                hybrid_concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)),
                hybrid_concat_preds(bbox_preds))


class BaseNetwork(nn.HybridBlock):  # VGG base network, without fc
    def __init__(self, IF_TINY, **kwargs):
        super(BaseNetwork, self).__init__(**kwargs)
        self.IF_TINY = IF_TINY
        self.conv1_1 = nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu')
        self.conv1_2 = nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu')
        self.pool1 = nn.MaxPool2D(pool_size=(2, 2))
        self.conv2_1 = nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu')
        self.conv2_2 = nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu')
        self.pool2 = nn.MaxPool2D(pool_size=(2, 2))
        self.conv3_1 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.conv3_2 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.conv3_3 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.pool3 = nn.MaxPool2D(pool_size=(2, 2))  # smaller here
        if not self.IF_TINY:
            self.conv4_1 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv4_2 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv4_3 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.pool4 = nn.MaxPool2D(pool_size=(2, 2))
            self.conv5_1 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv5_2 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv5_3 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.pool5 = nn.MaxPool2D(pool_size=(2, 2))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        if not self.IF_TINY:
            x = self.pool4(self.conv4_3(self.conv4_2(self.conv4_1(x))))
            x = self.pool5(self.conv5_3(self.conv5_2(self.conv5_1(x))))
        return x


def test(ctx=mx.cpu()):
    net = MySSD(1, num_anchors)
    net.initialize(init="Xavier", ctx=ctx)
    net.hybridize()
    # print(net)
    # x = nd.random.normal(0,1,(100,3,256,256), ctx=ctx)
    # net(x)
    batch_size, edge_size = 4, 256
    train_iter, _ = predata.load_data_uav(batch_size, edge_size)
    batch = train_iter.next()
    batch.data[0].shape, batch.label[0].shape

    if batch_size >= 25:  # show fucking pikachuus in grid
        imgs = (batch.data[0][0:25].transpose((0, 2, 3, 1))) / 255
        axes = utils.show_images(imgs, 5, 5).flatten()
        for ax, label in zip(axes, batch.label[0][0:25]):
            utils.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

        plt.show()

    # net.initialize(init=init.Xavier(), ctx=ctx)
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd',
                               {'learning_rate': 0.2, 'wd': 5e-4})
    cls_loss = gloss.SoftmaxCrossEntropyLoss()
    bbox_loss = gloss.L1Loss()

    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        cls = cls_loss(cls_preds, cls_labels)
        bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
        return cls + bbox

    def cls_eval(cls_preds, cls_labels):
        # the result from class prediction is at the last dim
        # argmax() should be assigned with the last dim of cls_preds
        return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()

    def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
        return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()

    IF_LOAD_MODEL = args.load
    if IF_LOAD_MODEL:
        net.load_parameters(args.model_path)
    else:
        for epoch in range(10):
            acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
            train_iter.reset()  # reset data iterator to read-in images from beginning
            start = time.time()
            for batch in train_iter:
                X = batch.data[0].as_in_context(ctx)
                Y = batch.label[0].as_in_context(ctx)
                with autograd.record():
                    # generate anchors and generate bboxes
                    im, anchors, cls_preds, bbox_preds = net(X)
                    # assign classes and bboxes for each anchor
                    bbox_labels, bbox_masks, cls_labels = nd.contrib.MultiBoxTarget(anchors, Y,
                                                                                    cls_preds.transpose((0, 2, 1)))
                    # calc loss
                    l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                                  bbox_masks)
                l.backward()
                trainer.step(batch_size)
                acc_sum += cls_eval(cls_preds, cls_labels)
                n += cls_labels.size
                mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
                m += bbox_labels.size

            if (epoch + 1) % 1 == 0:
                print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                    epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))

    net.save_parameters("myssd.params")

    def predict(X):
        im, anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
        # im = im.transpose((2, 3, 1, 0)).asnumpy()
        # imgs = [im[1:-2,1:-2, k, 0] for k in range(256)] # why are there boundary effect?

        # utils.show_images_np(imgs, 16, 16)
        # plt.show()
        # plt.savefig("./activation/figbase%s"%nd.random.randint(0,100,1).asscalar())

        # plt.imshow(nd.sum(nd.array(im[1:-2, 1:-2, :, :]), axis=2).asnumpy()[:, :, 0], cmap='gray')
        # plt.savefig("./suming_act")
        cls_probs = cls_preds.softmax().transpose((0, 2, 1))
        output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
        if idx == []:
            raise ValueError("NO TARGET. Seq Terminated.")
        return output[0, idx]

    def display(img, output, threshold):
        lscore = []
        for row in output:
            lscore.append(row[1].asscalar())
        for row in output:
            score = row[1].asscalar()
            if score < min(max(lscore), threshold):
                continue
            h, w = img.shape[0:2]
            bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
            cv.rectangle(img, (bbox[0][0].asscalar(), bbox[0][1].asscalar()),
                         (bbox[0][2].asscalar(), bbox[0][3].asscalar()), (0, 255, 0), 3)
            cv.imshow("res", img)
            cv.waitKey(60)

    cap = cv.VideoCapture(args.test_path)
    rd = 0
    while True:
        ret, frame = cap.read()
        img = nd.array(frame)
        feature = image.imresize(img, 256, 256).astype('float32')
        X = feature.transpose((2, 0, 1)).expand_dims(axis=0)

        countt = time.time()
        output = predict(X)
        if rd == 0: net.export('ssd')
        countt = time.time() - countt
        print("SPF: %3.2f" % countt)

        utils.set_figsize((5, 5))

        display(frame / 255, output, threshold=0.8)
        plt.show()
        rd += 1


test(mx.gpu())
