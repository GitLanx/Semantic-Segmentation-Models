import numpy as np
import tensorflow as tf
import os
import argparse
from sklearn.model_selection import train_test_split
from utils import get_dataset, generate_images
from metrics import mean_iou
from model_loader import load_model

tf.enable_eager_execution()
np.random.seed(1)
tf.set_random_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', type=str, default='FCN32', help='Model to train for')

# parameters
classes = 14
batch_size = 4
epochs = 1
palette = list(range(classes))
resized_shape = [96, 96]
plots = 4

# choose model to train for
args = parser.parse_args()
model = load_model(args.model, resized_shape, classes)

# tf.keras.utils.plot_model(model, to_file=args.model + '.png')

# dataset directory
data_dir = r'C:\Users\lan\Desktop\毕业论文\数据集\数据库A(200)\FS_400x300/'
mask_dir = r'C:\Users\lan\Desktop\毕业论文\数据集\数据库A(200)\label_png/'

# load dataset
data = os.listdir(data_dir)
data = [os.path.join(data_dir, data_) for data_ in data]
mask = os.listdir(mask_dir)
mask = [os.path.join(mask_dir, mask_) for mask_ in mask]

train_data, val_data, train_mask, val_mask = train_test_split(
    data, mask, test_size=0.2)

train_dataset = get_dataset(train_data, train_mask, palette,
                            resized_shape).batch(batch_size)
val_dataset = get_dataset(val_data, val_mask, palette,
                          resized_shape).batch(batch_size)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True)
]

model.compile(
    optimizer=tf.train.AdamOptimizer(0.0001),
    loss='categorical_crossentropy',
    metrics=[mean_iou])
model.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=len(data) // batch_size,
    validation_data=val_dataset,
    validation_steps=20,
    callbacks=callbacks)

for image, label in val_dataset.take(1):
    generate_images(model, image, label, plots=plots)
