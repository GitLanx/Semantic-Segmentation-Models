import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
import numpy as np


class Metric:
    """
    adapted from:
    https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.conf_mat = np.ndarray((num_classes, num_classes), dtype=np.int32)

    def accuracy(self, y_true, y_pred):
        """
        return mean class accuracy excluding void class
        """
        return tf.compat.v1.py_func(self._cal_per_cls_acc, [y_true, y_pred],
                          [tf.float32])

    def iou(self, y_true, y_pred):
        """
        return mean iou excluding void class
        """
        return tf.compat.v1.py_func(self._cal_miou, [y_true, y_pred], [tf.float32])

    def _cal_per_cls_acc(self, y_true, y_pred):
        self._cal_conf_mat(y_true, y_pred)
        per_cls_acc = np.diag(self.conf_mat) / (self.conf_mat.sum(1) + 1e-10)
        return np.nanmean(per_cls_acc[1:]).astype(np.float32)

    def _cal_miou(self, y_true, y_pred):
        self._cal_conf_mat(y_true, y_pred)
        true_positive = np.diag(self.conf_mat)
        false_positive = np.sum(self.conf_mat, 0) - true_positive
        false_negative = np.sum(self.conf_mat, 1) - true_positive

        iou = true_positive / (
            true_positive + false_positive + false_negative + 1e-10)
        return np.nanmean(iou[1:]).astype(np.float32)

    def _cal_conf_mat(self, y_true, y_pred):
        # 0 for the void class, excluded when evaluation
        y_pred = np.round(y_pred)
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        self.conf_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=self.num_classes)
        self.conf_mat = np.asarray(self.conf_mat)
        self.conf_mat[:, 0] = 0
        self.conf_mat[0, :] = 0


class ClassIoU(Callback):
    def __init__(self, x, num_classes, take):
        Callback.__init__(self)
        self.x = x
        self.num_classes = num_classes
        self.take = take

    def on_train_end(self, logs=None):
        confusion_matrix = tf.cast(
            tf.zeros([self.num_classes, self.num_classes]), tf.int32)
        for x, y_true in self.x.take(self.take):
            y_pred = self.model.predict(x)
            y_true = tf.reshape(tf.argmax(input=y_true, axis=-1), [-1])
            y_pred = tf.reshape(tf.argmax(input=y_pred, axis=-1), [-1])
            confusion_matrix = tf.add(
                confusion_matrix,
                tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=self.num_classes))

        confusion_matrix = np.asarray(confusion_matrix)
        confusion_matrix[:, 0] = 0
        confusion_matrix[0, :] = 0

        true_positive = np.diag(confusion_matrix)
        false_positive = np.sum(confusion_matrix, 0) - true_positive
        false_negative = np.sum(confusion_matrix, 1) - true_positive

        iou = true_positive / (
            true_positive + false_positive + false_negative + 1e-10)
        iou = np.around(iou, decimals=4)
        print("class IoU:", iou[1:])
        print('mIoU:', np.nanmean(iou[1:]))
