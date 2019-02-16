import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import generate_images, get_dataset
from metrics import Metric, ClassIoU
from model_loader import load_model

tf.enable_eager_execution()
np.random.seed(1)
tf.set_random_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UNet', help='Model to train for')
parser.add_argument('--train_data', type=str, default='', help='Path to training images')
parser.add_argument('--train_labels', type=str, default='', help='Path to training labels')
parser.add_argument('--val_data', type=str, default=None, help='Path to validation images, if not specified, val_data will sample from train_data')
parser.add_argument('--val_labels', type=str, default=None, help='Path to validation labels, if not specified, val_labels will sample from train_labels')
parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_classes', type=int, default=14, help='Number of classes, including void class or background')
parser.add_argument('--resized_height', type=int, default=96, help='Height of resized input')
parser.add_argument('--resized_width', type=int, default=96, help='Width of resized input')
# parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation')

args = parser.parse_args()
n_classes = args.n_classes
batch_size = args.batch_size
resized_shape = [args.resized_height, args.resized_width]
train_data = args.train_data
train_labels = args.train_labels
val_data = args.val_data
val_labels = args.val_labels
iou = Metric(n_classes).iou
acc = Metric(n_classes).accuracy

model = load_model(args.model, resized_shape, n_classes)

# tf.keras.utils.plot_model(model, to_file=args.model + '.png')

# load dataset
if val_data is None:
    train_data = os.listdir(train_data)
    train_data = [os.path.join(args.train_data, data) for data in train_data]
    train_labels = os.listdir(train_labels)
    train_labels = [os.path.join(args.train_labels, label) for label in train_labels]

    # Set random_state to make sure models are validated on the same validation images.
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=1234)

else:
    train_data = os.listdir(train_data)
    train_data = [os.path.join(args.train_data, data) for data in train_data]
    train_labels = os.listdir(train_labels)
    train_labels = [os.path.join(args.train_labels, label) for label in train_labels]
    val_data = os.listdir(val_data)
    val_data = [os.path.join(args.val_data, data) for data in val_data]
    val_labels = os.listdir(val_labels)
    val_labels = [os.path.join(args.val_labels, label) for label in val_labels]

train_dataset = get_dataset(
    train_data, train_labels, n_classes, batch_size, split='train',
    resized_shape=resized_shape)
val_dataset = get_dataset(
    val_data, val_labels, n_classes, batch_size, split='val',
    resized_shape=resized_shape)

iou_callback = ClassIoU(val_dataset, n_classes, len(val_data) // batch_size)

callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    # tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: 0.0001 + 0.02 * 0.5**(1 + epoch), verbose=True),
    # tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True),
    iou_callback
]

model.compile(
    optimizer=tf.train.AdamOptimizer(0.0001),
    loss='categorical_crossentropy',
    metrics=[iou, acc])

History = model.fit(
    train_dataset,
    epochs=args.epochs,
    steps_per_epoch=len(train_data) // batch_size,
    validation_data=val_dataset,
    validation_steps=len(val_data) // batch_size,
    callbacks=callbacks)

plt.figure(figsize=(6, 6))
plt.subplot(211)
plt.plot(History.history['loss'], label='train_loss')
plt.plot(History.history['val_loss'], label='val_loss')
plt.title('train loss vs validation loss ')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(212)
plt.plot(History.history['iou'], label='train_iou')
plt.plot(History.history['val_iou'], label='val_iou')
plt.title('train iou vs validation iou')
plt.ylabel('mIoU')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.show()

for image, label in val_dataset.take(1):
    generate_images(model, image, label, plots=4)
