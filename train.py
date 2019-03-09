import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import generate_images, get_dataset
from metrics import Metric, ClassIoU
from model_loader import load_model
from Dataloader import get_loader

np.random.seed(1)
tf.random.set_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UNet', help='Model to train for')
parser.add_argument('--train_data', type=str, default=r'D:\lx\Camvid\train', help='Path to training images')
parser.add_argument('--train_labels', type=str, default=r'D:\lx\Camvid\trainannot', help='Path to training labels')
parser.add_argument('--val_data', type=str, default=r'D:\lx\Camvid\val', help='Path to validation images, if not specified, val_data will sample from train_data')
parser.add_argument('--val_labels', type=str, default=r'D:\lx\Camvid\valannot', help='Path to validation labels, if not specified, val_labels will sample from train_labels')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_classes', type=int, default=12, help='Number of classes, including void class or background')
parser.add_argument('--img_size', type=tuple, default=(96, 96), help='resize images to proper size')
# parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation')

args = parser.parse_args()

n_classes = args.n_classes
batch_size = args.batch_size
train_data = args.train_data
train_labels = args.train_labels
val_data = args.val_data
val_labels = args.val_labels
iou = Metric(n_classes).iou
acc = Metric(n_classes).accuracy

loader = get_loader('camvid')
train_dataset = loader(r'D:\lx\Camvid', 'train', img_size=args.img_size).get_dataset(args.batch_size)
val_dataset = loader(r'D:\lx\Camvid', 'val', img_size=args.img_size).get_dataset(args.batch_size)

model = load_model(args.model, args.img_size, n_classes)

# tf.keras.utils.plot_model(model, to_file=args.model + '.png')

# load dataset
iou_callback = ClassIoU(val_dataset, n_classes, len(val_data) // batch_size)

callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    # tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: 0.0001 + 0.02 * 0.5**(1 + epoch), verbose=True),
    # tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True),
    iou_callback
]

model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(0.0001),
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=[tf.metrics.MeanIoU(args.n_classes)])

History = model.fit(
    train_dataset,
    epochs=args.epochs,
    steps_per_epoch=len(train_data) // batch_size,
    validation_data=val_dataset,
    validation_steps=len(val_data) // batch_size,
    # callbacks=callbacks,
)

plt.figure(figsize=(6, 6))
plt.subplot(211)
plt.plot(History.history['loss'], label='train_loss')
plt.plot(History.history['val_loss'], label='val_loss')
plt.title('train loss vs validation loss ')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(212)
plt.plot(History.history['mean_io_u'], label='train_iou')
plt.plot(History.history['val_mean_io_u'], label='val_iou')
plt.title('train iou vs validation iou')
plt.ylabel('mIoU')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.show()

for image, label in val_dataset.take(1):
    generate_images(model, image, label, plots=1)
