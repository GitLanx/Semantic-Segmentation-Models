import tensorflow as tf
from metrics import dice_coef


def discriminator_loss(y_true, y_pred):
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(y_true), logits=y_true)
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(y_pred), logits=y_pred)

    total_loss = real_loss + generated_loss

    return total_loss


def adv_gen_loss(y_true, y_pred, disc_output):
    generated_loss = tf.losses.softmax_cross_entropy(y_true, y_pred)
    disc_loss = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(disc_output), disc_output)
    # l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # generated_loss = tf.losses.sigmoid_cross_entropy(
    #     tf.ones_like(disc_output), disc_output)
    # return generated_loss + 100 * l1_loss

    return generated_loss + 0.5 * disc_loss


def generator_loss(y_true, y_pred):
    return tf.losses.softmax_cross_entropy(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
