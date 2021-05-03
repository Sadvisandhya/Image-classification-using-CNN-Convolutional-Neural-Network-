# Image-classification-using-CNN-Convolutional-Neural-Network-

Dataset

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Content
Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.
Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.

The training and test data sets have 785 columns.

The first column consists of the class labels (see above), and represents the article of clothing.
The rest of 784 columns (1-785) contain the pixel-values of the associated image.

Read the data

There are 10 different classes of images, as following:

•	0: T-shirt/top;
•	1: Trouser;
•	2: Pullover;
•	3: Dress;
•	4: Coat;
•	5: Sandal;
•	6: Shirt;
•	7: Sneaker;
•	8: Bag;
•	9: Ankle boot.

Image dimensions are 28x28.


Train the model

Build the model

We used a Sequential(CNN) model.

•	The Sequential model is a linear stack of layers. It can be first initialized and then we add layers using add method or we can add all layers at initial stage. The layers added are as follows:
•	Conv2D is a 2D Convolutional layer (i.e. spatial convolution over images). The parameters used are:

	filters - the number of filters (Kernels) used with this layer; here filters = 32;

	kernel_size - the dimmension of the Kernel: (3 x 3);

	activation - is the activation function used, in this case relu;

	kernel_initializer - the function used for initializing the kernel;

	input_shape - is the shape of the image presented to the CNN: in our case is 28 x 28 The input and output of the Conv2D is a 4D tensor.

•	MaxPooling2D is a Max pooling operation for spatial data. Parameters used here are:

	pool_size, in this case (2,2), representing the factors by which to downscale in both directions;

	Conv2D with the following parameters:
	filters: 64;
	kernel_size : (3 x 3);
	activation : relu;

•	Conv2D with the following parameters:
	filters: 128;
	kernel_size : (3 x 3);
                         

The validation accuracy does not improve after few epochs and the validation loss is increasing after few epochs. This confirms our assumption that the model is overfitted. We tried to improve the model by adding Dropout layers.

Add Dropout layers to the model

We add several Dropout layers to the model, to help avoiding overfitting.

Dropout is helping avoid overfitting in several ways, as explained in [6] and [7].
	
	activation : relu;
•	Flatten. This layer Flattens the input. Does not affect the batch size. It is used without parameters;
•	Dense. This layer is a regular fully-connected NN layer. It is used without parameters;

	units - this is a positive integer, with the meaning: dimensionality of the output space; in this case is: 128;

	activation - activation function : relu;
•	Dense. This is the final layer (fully connected). It is used with the parameters:

	units: the number of classes (in our case 10);
	activation : softmax; for this final layer it is used softmax activation (standard for multiclass classification)
The last layer is a 10-node softmax layer that returns an array of 10 probability scores that sum to 1. Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.

Because we are dealing with the classification problem, the final layer uses softmax activation to get class probabilities. As class probabilities follow a certain distribution, cross-entropy indicates the distance from networks preferred distribution.

Then we compile the model, specifying as well the following parameters:

•	loss;
•	optimizer;
•	metrics.

•	After adding the Dropout layers, the validation accuracy and validation loss are much better. 

Two plots which illustrate the accuracy / loss of training and validation over the time:

![alt text](https://github.com/Sadvisandhya/Image-classification-using-CNN-Convolutional-Neural-Network-/blob/main/mod_acc_1.png?raw=true)

Conclusions
With a complex sequential model with multiple convolution layers and 40 epochs for the training, we obtained an accuracy 0.937 for test prediction. We trained the model with Dropout layers to reduce overfitting.
Only few classes are not correctly classified all the time, especially Class 6 (Shirt) and Class 4(coat).
