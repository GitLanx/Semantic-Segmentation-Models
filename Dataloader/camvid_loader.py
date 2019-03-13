import tensorflow as tf
import os
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CamVidLoader:
    class_names = np.array([
        'sky',
        'building',
        'pole',
        'road',
        'pavement',
        'tree',
        'sign',
        'fence',
        'vehicle',
        'pedestrian',
        'bicyclist',
        'void'
    ])

    def __init__(
        self,
        root,
        split="train",
        img_size=None,
        augmentations=None
    ):
        self.root = root
        self.split = split
        self.augmentations = augmentations
        self.n_classes = 12
        self.img_size = img_size

        path = os.path.join(self.root, self.split + ".txt")
        with open(path, "r") as f:
            self.file_list = [file_name.rstrip() for file_name in f]

        self.shuffle_size = len(self.file_list)

        print(f"Found {len(self.file_list)} {split} images")

    def __len__(self):
        return len(self.file_list)

    def get_dataset(self, batch_size):
        img_name = [name.split()[0].split('/')[-1] for name in self.file_list]
        img_path = [os.path.join(self.root, self.split, name) for name in img_name]
        if self.split == 'train':
            lbl_path = [os.path.join(self.root, 'trainannot', name) for name in img_name]
        elif self.split == 'val':
            lbl_path = [os.path.join(self.root, 'valannot', name) for name in img_name]
        elif self.split == 'test':
            lbl_path = [os.path.join(self.root, 'testannot', name) for name in img_name]

        dataset = tf.data.Dataset.from_tensor_slices((img_path, lbl_path))
        if self.split == 'train':
            dataset = dataset.shuffle(self.shuffle_size)
            dataset = dataset.map(self.parse_function, num_parallel_calls=4)
        elif self.split == 'val' or 'test':
            dataset = dataset.map(self.parse_function, num_parallel_calls=4)

        # if self.augmentations is not None:
        #     img, lbl = self.augmentations(img, lbl)

        return dataset.batch(batch_size).prefetch(batch_size)

    def parse_function(self, x, y):
        image = tf.io.read_file(x)
        image = tf.cond(
            pred=tf.image.is_jpeg(image),
            true_fn=lambda: tf.image.decode_jpeg(image),
            false_fn=lambda: tf.image.decode_png(image))
        image = tf.cast(image, tf.float32)

        label = tf.io.read_file(y)
        label = tf.image.decode_png(label)

        if self.img_size:
            image = tf.image.resize(
                image,
                size=[self.img_size[0], self.img_size[1]],
                method=tf.image.ResizeMethod.BILINEAR)
            label = tf.image.resize(
                label,
                size=[self.img_size[0], self.img_size[1]],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        label = tf.squeeze(label, axis=-1)
        label = tf.one_hot(label, depth=self.n_classes)

        return image / 255.0, label
