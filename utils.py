import tensorflow as tf
import matplotlib.pyplot as plt


def get_dataset(images,
                labels,
                n_classes,
                dtype,
                resized_shape=[96, 96],
                palette=None):
    """Use tf.data.Dataset to read image files.

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
        lambda x, y: parse_function(x, y, n_classes, resized_shape, palette))
    if dtype == 'train':
        # dataset = tf.data.Dataset.zip((images,
        #                                labels)).shuffle(shuffle_size).repeat()
        dataset = dataset.repeat()
        return dataset
    elif dtype == 'val' or 'test':
        # dataset = tf.data.Dataset.zip((images, labels))
        return dataset


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


def parse_function(filename, label, n_classes, resized_shape, palette):
    """
    function for parse images and labels
    """
    images = load_image(filename, resized_shape)
    labels = load_label(filename, n_classes, resized_shape, palette)
    return images, labels


def one_hot_encode(label, n_classes, palette):
    """change labels to one-hot encoding.

    :param label: labels
    :param palette: use self-defined label if specified
    :returns: one-hot encoded labels
    """
    if palette is None:
        label = tf.squeeze(label, axis=-1)
        one_hot_map = tf.one_hot(label, n_classes)
    else:
        one_hot_map = []
        for colour in palette:
            class_map = tf.reduce_all(tf.equal(label, colour), axis=-1)
            one_hot_map.append(class_map)

        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

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


def load_label(filename, n_classes, resized_shape, palette):
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
    label = one_hot_encode(label, n_classes, palette)

    return label
