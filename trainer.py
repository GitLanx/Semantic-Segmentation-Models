import datetime
import os
import os.path as osp
import shutil
from utils import label_accuracy_score, visualize_segmentation, get_tile_image
import numpy as np
import pytz
import scipy.misc
import tqdm
import tensorflow as tf


class Trainer:
    def __init__(self, model, optimizer, train_dataset, val_dataset,
                 train_size, val_size, out, epochs, n_classes, val_epoch=None):

        self.model = model
        self.optim = optimizer

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_size = train_size
        self.val_size = val_size

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('UTC'))

        self.val_epoch = val_epoch

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.n_classes = n_classes
        self.epoch = 0
        self.epochs = epochs
        self.best_mean_iu = 0

    # @tf.function
    def train_epoch(self):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        cross_entropy = tf.losses.CategoricalCrossentropy()

        metrics = []

        for data, target in tqdm.tqdm(
                self.train_dataset, total=self.train_size,
                desc=f'Train epoch={self.epoch}', ncols=80, leave=False):

            with tf.GradientTape() as tape:
                score = self.model(data, training=True)
                loss = cross_entropy(target, score)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))

            # epoch_loss_avg(target, score)
            # epoch_accuracy(target, score)

            target = target.numpy()
            score = score.numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                label_accuracy_score(
                    np.argmax(target, -1), np.argmax(score, -1), self.n_classes)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))

        metrics = np.mean(metrics, axis=0)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('UTC')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch] + [loss.numpy()] + \
                metrics.tolist() + [''] * 5 + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        if self.epoch % self.val_epoch == 0:
            self.validate()

    # @tf.function
    def validate(self):
        global val_size
        cross_entropy = tf.losses.CategoricalCrossentropy()
        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for data, target in tqdm.tqdm(
                self.val_dataset, total=self.val_size,
                desc=f'Valid epoch={self.epoch}', ncols=80, leave=False):

            score = self.model(data, training=False)
            loss = cross_entropy(target, score)

            val_loss += loss.numpy()
            target = target.numpy()
            score = score.numpy()
            for img, lt, lp in zip(data, np.argmax(target, -1), np.argmax(score, -1)):
                label_trues.append(lt)
                label_preds.append(lp)
                # if len(visualizations) < 9:
                #     viz = visualize_segmentation(
                #         lbl_pred=lp, lbl_true=lt, img=img,
                #         n_class=self.n_classes)
                #     visualizations.append(viz)
        metrics = label_accuracy_score(
            label_trues, label_preds, self.n_classes)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'epoch%08d.jpg' % self.epoch)
        # scipy.misc.imsave(out_file, get_tile_image(visualizations))

        val_loss /= self.val_size

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('UTC')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
            self.model.save(osp.join(self.out, 'best_model.h5'))

        # if is_best:
            # shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
            #             osp.join(self.out, 'model_best.pth.tar'))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.epochs + 1,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
