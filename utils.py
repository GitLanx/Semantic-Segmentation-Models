import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def data_loader(train_data, train_labels, val_data, val_labels, n_classes,
                batch_size, resized_shape):
    '''load data
    
    :param train_data: root of training data
    :param train_labels: root of training labels
    :param val_data: root of validation data, if None, take 20% of train_data as val_data
    :param val_labels: root of validation labels
    :param n_classes: number of classes
    :param batch_size: number of images in each batch
    :param resized_shape: resize input data to proper size
    :returns: return tf.data.Dataset for training and validation'''
    if val_data is None:
        data = os.listdir(train_data)
        data = [os.path.join(train_data, _data) for _data in data]
        label = os.listdir(train_labels)
        label = [os.path.join(train_labels, _label) for _label in label]

        # Set random_state to make sure models are validated on the same validation images.
        train_data, val_data, train_label, val_label = train_test_split(
            data, label, test_size=0.2, random_state=1234)

        train_dataset = get_dataset(
            train_data, train_label, n_classes, batch_size,
            resized_shape=resized_shape)
        val_dataset = get_dataset(
            val_data, val_label, n_classes, batch_size,
            resized_shape=resized_shape)
        return train_dataset, val_dataset
    else:
        train_data = os.listdir(train_data)
        train_data = [os.path.join(train_data, data) for data in train_data]
        train_label = os.listdir(train_label)
        train_label = [os.path.join(train_label, label) for label in train_label]
        val_data = os.listdir(val_data)
        val_data = [os.path.join(val_data, data) for data in val_data]
        val_label = os.listdir(val_label)
        val_label = [os.path.join(val_label, label) for label in val_label]
        
        train_dataset = get_dataset(
            train_data, train_label, n_classes, batch_size,
            resized_shape=resized_shape)
        val_dataset = get_dataset(
            val_data, val_label, n_classes, batch_size,
            resized_shape=resized_shape)
        return train_dataset, val_dataset


def get_dataset(images, labels, n_classes, batch_size, resized_shape=[96, 96]):
    """Use tf.data.Dataset to read image and labels.

    :param images: list of image filenames
    :param labels: list of label filenames
    :param n_classes: number of classes, including void class
    :param resized_shape: rescale images to proper shape
    :param palette: label pixel for each class, if you have special labels,
                    specify them in a list
    :returns: return a tf.data.Dataset
    """
    assert type(images) and type(
        labels) is list, 'Type of images and labels must be list'

    shuffle_size = len(images)
    # images = tf.data.Dataset.list_files(images, shuffle=False)
    # images = images.map(
    #     lambda x: load_image(x, resized_shape), num_parallel_calls=4)

    # labels = tf.data.Dataset.list_files(labels, shuffle=False)
    # labels = labels.map(
    #     lambda x: load_label(x, classes, resized_shape, palette),
    #     num_parallel_calls=4)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.map(
        lambda x, y: parse_function(x, y, n_classes, resized_shape),
        num_parallel_calls=4)
    return dataset.batch(batch_size)


def generate_images(model, input_image, target_image, plots=1):
    """plot input_image, target_image and prediction in one row, all with
    shape [batch_size, height, width, channels].

    :param model: trained model
    :param input_image: a batch of input images
    :param target_image: a batch of target images
    :param plots: numbers of image groups you want to plot, default 1
    """
    assert plots <= input_image.shape[
        0], "plots number should be less than batch size"

    classes = target_image.shape[-1].value
    prediction = model.predict(input_image)
    plt.figure(figsize=(20, 20))

    target_image = tf.argmax(target_image, axis=-1)
    prediction = tf.argmax(prediction, axis=-1)

    for i in range(plots):
        plt.subplot(plots, 3, i * 3 + 1)
        plt.imshow(input_image[i], vmin=0, vmax=classes)
        plt.subplot(plots, 3, i * 3 + 2)
        plt.imshow(target_image[i], vmin=0, vmax=classes)
        plt.subplot(plots, 3, i * 3 + 3)
        plt.imshow(prediction[i], vmin=0, vmax=classes)
    plt.show()


def parse_function(images, labels, n_classes, resized_shape):
    """
    function for parse images and labels
    """
    images = load_image(images, resized_shape)
    labels = load_label(labels, n_classes, resized_shape)
    return images, labels


def one_hot_encode(label, n_classes):
    """change labels to one-hot encoding.

    :param label: labels
    :param palette: use self-defined label if specified
    :returns: one-hot encoded labels
    """
    label = tf.squeeze(label, axis=-1)
    one_hot_map = tf.one_hot(label, n_classes)

    # another one hot method using palette
    # one_hot_map = []
    # for colour in palette:
    #     class_map = tf.reduce_all(tf.equal(label, colour), axis=-1)
    #     one_hot_map.append(class_map)

    # one_hot_map = tf.stack(one_hot_map, axis=-1)
    # one_hot_map = tf.cast(one_hot_map, tf.float32)

    return one_hot_map


def load_image(filename, resized_shape):
    """load images.

    :param filename: image filenames
    :param resized_shape: resize the images to proper size
    :returns: images
    """
    image = tf.read_file(filename)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, size=resized_shape)

    return image


def load_label(filename, n_classes, resized_shape):
    """load labels.

    :param filename: label filenames
    :param palette: label pixel
    :param resized_shape: resize the labels to proper size
    :returns: one-hot encoded labels
    """
    label = tf.read_file(filename)
    label = tf.image.decode_png(label)
    label = tf.image.resize_images(
        label,
        size=resized_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = one_hot_encode(label, n_classes)

    return label
