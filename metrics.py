import tensorflow as tf
import tensorflow.keras.backend as K


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


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def mean_iou(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum((y_true + y_pred) - y_true * y_pred)
    return intersection / union


def pixel_accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    # y_true = K.around(y_true)
    # y_pred = K.around(y_pred)
    # return K.sum(y_true*y_pred)/K.sum(y_true)
    return tf.reduce_sum(y_true == y_pred) / y_true.shape


def class_accuracy(y_true, y_pred, num_classes):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    for i in range(num_classes):
        a = y_true == i
        b = y_pred == i
        acc = tf.reduce_sum(a * b) / (tf.reduce_sum(y_pred == i) + 1e-10)
        print('[' + str(acc) + ']', end='')
        print('')
