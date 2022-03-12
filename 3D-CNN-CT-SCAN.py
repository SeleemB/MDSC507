# 3D-CNN-CT-SCAN.py
# MDSC507
# Ahmed Almousawi and Seleem Badawy 


# Get necessary packages and libraries to begin neural network training.

import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# Downloading the MosMedData: Chest CT Scans with COVID-19 Related Findings.

# Download url of normal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
# Create a path to save the file to by combining the current working directory with the filename.
filename = os.path.join(os.getcwd(), "CT-0.zip")
# Gets the file from the URL and saves it.
keras.utils.get_file(filename, url) 

# Download url of abnormal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
# Create a path to save the file to by combining the current working directory with the filename.
filename = os.path.join(os.getcwd(), "CT-23.zip")
# Gets the file from the URL and saves it.
keras.utils.get_file(filename, url) 

# Make a new folder to store extracted data.
os.makedirs("MosMedData") 

# Unzips the data and puts it into the new folder.
with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

with zipfile.ZipFile("CT-23.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

    
# Loading data and preprocessing.    

# Import nibabel, which gives access to common medical and neuroimaging file formats.
import nibabel as nib

# From scipy, which provides algorithms for optimization, integration, interpolation, and many more statistical and mathematical problems, import ndimage, which various functions for multidimensional image processing.
from scipy import ndimage


def read_nifti_file(filepath):
    """Read and load volume from selected filepath"""
    # Read file.
    scan = nib.load(filepath)
    # Get raw data.
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize image across z-axis"""
    # Set the desired depth.
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth.
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor.
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate.
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis.
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan.
    volume = read_nifti_file(path)
    # Normalize.
    volume = normalize(volume)
    # Resize width, height and depth.
    volume = resize_volume(volume)
    return volume

# Folder "CT-0" consist of CT scans having normal lung tissue, no CT-signs of viral pneumonia.
# Create list of normal CT file paths.
normal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-0", x)
    for x in os.listdir("MosMedData/CT-0")
]
# Folder "CT-23" consist of CT scans having several ground-glass opacifications, involvement of lung parenchyma.
# Create list of normal CT file paths.
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-23", x)
    for x in os.listdir("MosMedData/CT-23")
]
# Display number of files for normal and abnormal CT scans.
print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))


# Build train and validate datasets.

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

# For the CT scans having presence of viral pneumonia.
# assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Split data in the ratio 70-30 for training and validation.
# Create 4 arrays; two for training the model and two for validating the data.
x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)


# Data augmentation.

import random

from scipy import ndimage

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    # Rotates the volumes by a semi-random angle in order to train the algorithm more effectively.

    def scipy_rotate(volume):
        """Determine the angles of ratation and rotate the volume"""
        # Define some rotation angles.
        angles = [-20, -10, -5, 5, 10, 20]
        # Pick angles at random.
        angle = random.choice(angles)
        # Rotate volume.
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume
    # Send the volume through the function we defined above.
    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)  
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume.
    volume = rotate(volume)
    # Adds a new dimension in order to facilitate 3D processing.
    volume = tf.expand_dims(volume, axis=3) 
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    # Adds a new dimension in order to facilitate 3D processing.
    volume = tf.expand_dims(volume, axis=3) 
    return volume, label

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the data during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

# Import matplotlib.pyplot for visualization.
import matplotlib.pyplot as plt

# Set variables necessary for visualization of an augmented CT scan.
data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]

# Display CT scan image.
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    # Adjust properties of data and set names for visualization.
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    # Set up and style subplots.
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])


# Define a 3D convolutional neural network.

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""
      # Input dimensions of scans into Keras.
    inputs = keras.Input((width, height, depth, 1))

    # Create a convolutional kernel with the layer input to produce a tensor of outputs.
    # Filters = 64 is the dimensionality or amount of output filters in the convolution.
    # Kernel_size is the size of the kernel matrix (depth, height, and width) that is used for blurring, sharpening, embossing, edge detection, and more.
    # Activation = "relu" is a piecewise linear function that will apply to the outputs if the input is positive, otherwise, it will output zero.
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    # Downscales the 3D input of each dimension by pool_size = 2 of x.
    x = layers.MaxPool3D(pool_size=2)(x)
    # Normalizes the data of batch x during training and inference to maintain the mean output close to 0 and output standard deviation close to 1.
    x = layers.BatchNormalization()(x)

    # Repeat the above three lines of code's procedure for 'x'.
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # Repeat the above three lines of code's procedure for 'x', but double the amount of output filters in the convolution.
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # Repeat the above three lines of code's procedure for 'x', but double the amount of output filters in the convolution.
    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # Apply Global Average Pooling function of convolutional neural networks to replace the fully connected layers generated.
    x = layers.GlobalAveragePooling3D()(x)
    # Dense the layers of x by a dimensionality of the "units" and with a relu activation.
    x = layers.Dense(units=512, activation="relu")(x)
    # Establish Dropout layer on x to prevent overfitting at a fraction of 0.3 units.
    x = layers.Dropout(0.3)(x)

    # Dense the layers of the final generated 'x' by a dimensionality of 1 and with an activation of 'sigmoid', which guarantees that output of the unit is between 0 and 1.
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()

# Train model.

# Compile model.
initial_learning_rate = 0.0001  # Sets initial learning rate.
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    # Uses exponential decay to lower the learning rate as time progresses.
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), # Sets the "Adam" optimizer.
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
) # Saves the best model for later use.
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)


# Visualizing model performance.

# Creates two subplots to graph upon.
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

# Uses MatPlotLib to graph the accuracy vs time of the model and the loss vs time.
for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

    
# Make predictions on a single CT scan.

# Load best weights.
model.load_weights("3d_image_classification.h5")  # Gets the "best" model from before and uses it to make our prediction.
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]] # Gets our normal/abnormal scores.

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):  # Prints the scores for normal and abnormal weightings.
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )
