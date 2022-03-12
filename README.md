# MDSC507

3D Image Classification From CT Scans
Seleem and Ahmed

The 3D Image Classification From CT Scans system utilizes Tensorflow’s Keras deep learning API, to produce a supervised machine learning 3D convolutional neural network (CNN) algorithm that is taught to recognize the presence of viral pneumonia in CT scans. The model, after training, using kernels to process the datasets in this CNN, looks for patterns in the CT scans and predicts if viral pneumonia may exist in the lungs by identifying normal and abnormal CT scans. First, the appropriate libraries and packages are installed.

Installation and Datasets

The program relies on several libraries and modules that must be installed prior to running the program. If one or more of these libraries is not present, the program will produce errors and not perform the task effectively. The libraries and modules and their use cases are listed as such:
Library/Module
Usage
Installation
OS
Installs files from an external source and saves them onto the computer.
N/A (Default Module)
Zipfile
Takes the downloaded .zip files and decompresses and extracts the date from them.
N/A (Default Module)
Numpy
Creates arrays and uses them to organize and manipulate the data
pip install numpy
Tensorflow
Contains the neural network framework, as well as the Keras API used in the program. Creates the convolutional neural network.
pip install tensorflow
Nibabel
Provides access to the CT scan files.
pip install nibabel
SciPy
Rotates and resizes the CT scan files into a consistent format.
pip install scipy
Random
Randomizes the degrees that the images are rotated by to further train the neural network
N/A (Default Module)
MatPlotLib
Creates graphs depicting the accuracy vs time and the loss vs time.
pip install matplotlib


The files that are processed by the code are taken as a subset of data from the MosMedData Chest CT scan dataset. The full dataset, if needed, can be downloaded here. [https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1]. The program automatically downloads the subset of data from Github and extracts it into the current working directory. If needed, the subsets can be manually downloaded here:
Normal CT Scans [https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip]
Abnormal CT Scans [https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip]

Program Overview
Following pre-processing, with the more easily accessible numbers and data points for the computer, the datasets are further processed and validated prior to training the neural network. First, each scan is resized and rescaled, followed by labeling of the two conditions (normal and abnormal). The data is split and concatenated into 4 arrays for training and validation.

Data loaders to train and validate the data are defined and the data is augmented by rotating the data points at semi-random angles to improve the training and understanding of the CT scan data and images. A new dimension is also added to facilitate 3D processing. 

The program proceeds to use MatPlotLib to visualize an augmented CT scan followed by a CT scan displaying each of a series of slices. This helps the user to understand what the CT scans look like and how they have been processed to be received by the computer for training and analysis.

The 3D convolutional neural network is then defined and the final processing is done prior to training 3 times. This includes creating the convolutional kernels with a layer input to produce a tensor of outputs; downscaling; and normalization of batch data. Following this, global average pooling, layer densing, and layer dropout are applied to finalize and condense the code to then return the CNN model.

After setting the learning rate and defining the callback, the model is trained for 15 epochs using the functions and actions defined previously, with validation after each epoch. The model accuracy and loss for the training and validation sets are then plotted on two subplots to visualize the unbiased model performance and accuracy. 

Finally, the model receives a single CT scan and uses the best model defined during training and makes predictions on it, displaying the final output to the user of the normal and abnormal confidence scores of said prediction as shown:

“ This model is 3.66 percent confident that CT scan is normal
This model is 96.34 percent confident that CT scan is abnormal .”

Final outputs
The visualizations of the CT scans and CT scan slices.


The graphs of the unbiased model performance: model accuracy and loss for the training and validation sets.
The analysis and results of confidence scores when analyzing a single CT scan, how confident the program is that the scan is normal and. abnormal

Known Potential Issues of Code
One of known issues with the program is that the program’s OS module doesn’t check to see if the necessary files exist already. Thus, after the initial full run, it is unable to move past the “Downloading the MosMedData: Chest CT Scans with COVID-19 Related Findings” section, as the OS module cannot create a directory with the same name as the existing one created from the first run. To remedy this when re-running the code, either comment out this section, delete the downloaded files prior to rerunning the program, or simply run everything after this chunk of code, which should run successfully.

Attributions
The code featured in this program is by Hasib Zunair, created for the keras.io website in order to showcase the creation of a 3D convolutional neural network. The source can be accessed through the website (https://keras.io/examples/vision/3D_image_classification/) or the Github (https://github.com/keras-team/keras-io/blob/master/examples/vision/3D_image_classification.py) page
