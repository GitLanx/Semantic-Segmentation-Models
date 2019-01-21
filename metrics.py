import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
import numpy as np


def TP(y_true, y_pred):
    # True Positive (TP): predict a label of 1 (positive), and the true label is 1.
    return K.sum(tf.logical_and(y_true == 1, y_pred == 1))


def TN(y_true, y_pred):
    # True Negative (TN): predict a label of 0 (negative), and the true label is 0.
    return K.sum(tf.logical_and(y_true == 0, y_pred == 0))


def FP(y_true, y_pred):
    # False Positive (FP): predict a label of 1 (positive), but the true label is 0.
    return K.sum(tf.logical_and(y_true == 0, y_pred == 1))


def FN(y_true, y_pred):
    # False Negative (FN): predict a label of 0 (negative), but the true label is 1.
    return K.sum(tf.logical_and(y_true == 1, y_pred == 0))


def precision(y_true, y_pred):
    true_positive = K.sum(K.round(y_true * y_pred))
    predicted_positive = K.sum(K.round(y_pred))
    return true_positive / (predicted_positive + K.epsilon())


def recall(y_true, y_pred):
    true_positive = K.sum(K.round(y_true * y_pred))
    possible_positive = K.sum(K.round(y_true))
    return true_positive / (possible_positive + K.epsilon())


# equivalent to dice coefficient
def f1_score(y_true, y_pred):
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))


def dice_coef(y_true, y_pred):
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))


# include void class
# def mean_iou(y_true, y_pred):
#     # TP / (TP + FP + FN)
#     y_pred = K.round(y_pred)
#     intersection = K.sum(y_true * y_pred)
#     union = K.sum((y_true + y_pred) - y_true * y_pred)
#     return intersection / union


def class_accuracy(y_true, y_pred, num_classes):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    for i in range(num_classes):
        a = y_true == i
        b = y_pred == i
        acc = tf.reduce_sum(a * b) / (tf.reduce_sum(y_pred == i) + 1e-10)
        print('[' + str(acc) + ']', end='')
        print('')


class Metric:
    """
    adapted from:
    https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/utils/utils.py
    https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.conf_mat = np.ndarray((num_classes, num_classes), dtype=np.int32)

    def accuracy(self, y_true, y_pred):
        """
        return pixel accuracy
        """
        return tf.py_func(self._cal_pix_acc, [y_true, y_pred], [tf.float32])

    def iou(self, y_true, y_pred):
        """
        return mean iou excluding void class
        """
        return tf.py_func(self._cal_miou, [y_true, y_pred], [tf.float32])

    # def _cal_class_miou(self, y_true, y_pred):
    #     # a different way of calculate miou
    #     y_pred = np.round(y_pred)
    #     y_true = np.argmax(y_true, axis=-1)
    #     y_pred = np.argmax(y_pred, axis=-1)

    #     y_pred = y_pred * (y_true > 0)

    #     # Compute area intersection:
    #     intersection = y_pred * (y_pred == y_true)
    #     (area_intersection, _) = np.histogram(
    #         intersection, bins=self.num_classes, range=(1, self.num_classes))

    #     # Compute area union:
    #     (area_pred, _) = np.histogram(y_pred, bins=self.num_classes, range=(1, self.num_classes))
    #     (area_true, _) = np.histogram(y_true, bins=self.num_classes, range=(1, self.num_classes))
    #     area_union = area_pred + area_true - area_intersection

    #     return area_intersection / (area_union + 1e-10)

    def _cal_pix_acc(self, y_true, y_pred):
        self._cal_conf_mat(y_true, y_pred)
        overall_acc = np.diag(
            self.conf_mat).sum() / (self.conf_mat.sum() + 1e-10)
        return overall_acc.astype(np.float32)

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
        x = y_pred + self.num_classes * y_true
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        self.conf_mat = bincount_2d.reshape((self.num_classes,
                                             self.num_classes))
        self.conf_mat[:, 0] = 0
        self.conf_mat[0, :] = 0


class ClassIoU(Callback):
    def __init__(self, x, num_classes, val_take):
        super().__init__()

        self.x = x
        self.num_classes = num_classes
        self.val_take = val_take

    def on_train_end(self, logs=None):
        confusion_matrix = tf.cast(
            tf.zeros([self.num_classes, self.num_classes]), tf.int32)
        for x, y_true in self.x.take(self.val_take):
            y_pred = self.model.predict_on_batch(self.x)
            y_true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])
            y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
            confusion_matrix = tf.add(
                confusion_matrix,
                tf.confusion_matrix(y_true, y_pred, self.num_classes))

        confusion_matrix = np.asarray(confusion_matrix)
        confusion_matrix[:, 0] = 0
        confusion_matrix[0, :] = 0

        true_positive = np.diag(confusion_matrix)
        false_positive = np.sum(confusion_matrix, 0) - true_positive
        false_negative = np.sum(confusion_matrix, 1) - true_positive

        iou = true_positive / (
            true_positive + false_positive + false_negative + 1e-10)
        print("class IoU:", round(iou, 2))
