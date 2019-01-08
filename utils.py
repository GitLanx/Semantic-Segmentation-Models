import tensorflow as tf
import matplotlib.pyplot as plt


def one_hot_encode(label, palette):
    """change labels to one-hot encoding.

    :param label: labels
    :param palette: label pixel
    :returns: one-hot encoded labels
    """
    one_hot_map = []
    for colour in palette:
        class_map = tf.reduce_all(tf.equal(label, colour), axis=-1)
        one_hot_map.append(class_map)

    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)

    return one_hot_map


def load_image(file_name, resized_shape):
    """load images.

    :param file_name: image file names
    :param resized_shape: resized_shapeze the images to proper size
    :returns: images
    """
    image = tf.read_file(file_name)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, size=resized_shape)

    return image


def load_label(file_name, palette, resized_shape):
    """load labels.

    :param file_name: label file names
    :param palette: label pixel
    :param resized_shape: resized_shapeze the labels to proper size
    :returns: one-hot encoded labels
    """
    label = tf.read_file(file_name)
    label = tf.image.decode_png(label)
    label = tf.image.resize_images(
        label,
        size=resized_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = one_hot_encode(label, palette)

    return label


def get_dataset(images, labels, palette, resized_shape=[96, 96]):
    """Use tf.data.Dataset to read image files.

    :param images: list of image file names
    :param labels: list of label file names
    :param palette: label pixel for each class
    :param resized_shape: rescale images to proper shape
    :returns: return a tf.data.Dataset
    """
    shuffle_size = len(images)
    images = tf.data.Dataset.list_files(images, shuffle=False)
    images = images.map(
        lambda x: load_image(x, resized_shape), num_parallel_calls=4)

    labels = tf.data.Dataset.list_files(labels, shuffle=False)
    labels = labels.map(
        lambda x: load_label(x, palette, resized_shape), num_parallel_calls=4)

    dataset = tf.data.Dataset.zip((images,
                                   labels)).shuffle(shuffle_size).repeat()

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
    # prediction = model(input_image)
    plt.figure(figsize=(15, 15))

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
