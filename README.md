Student Name: Siyang Zhang



For this starter project, I build a neural network to do 3D shape classification.

First of all, I create my own dataset of 7 different basic 3D geometric shapes, including: ball, cube, cylinder, cone, pyramid(with a square base), ring, and torus. By calculating their curve functions, I tried to generate their corresponding 3D representation. For this project, I choose point cloud as the 3D representation.

Then I build a PointNet model according to the paper https://arxiv.org/abs/1612.00593 and documentation on Keras.

The PointNet classification network has two parts:

In the first part (I called it feature transform network), the network takes in a 2D array of unordered points, with shape n*3 (3 here represent for x, y, z on each axis, for simplicity). Then there is an input transform layer, mlp layer, a feature transform layer, and another mlp layer.

For each of the transform layer, the paper call it t-net, and it consist of 3 convolutional layers, with a max pooling layer, and some dense layers. The t-net finally combine both local information and global information to each point.

mlp layers are somehow tricky here. In the paper, the author suggested mlp works on both input and feature transformation, but on Keras, they use 1D convolution. Here I believe the reason is that we treat the point cloud as a 1D array of triples, and thus 1D convolution may extract useful features.

In the second part(I called aggregation network), a max-pooling layer first aggregate all information from these points together, making them order-irrelevant, and pass such data into another mlp, to get final probability distribution, which is a very basic classifier.



I trained the model for 20 epochs and it finally reached over 0.95 accuracy on a test dataset. In fact, the accuracy reached 1.0 on the 17th epoch, and thus the later may suffer from overfitting(the training record can be viewed in the jupyter notebook file).

I believe the biggest challenge in this project is the t-net part. The paper suggest that mlp is used in input and feature transform as feature extractor, but it turn out that 1D convolution perform better. Also, the problem is 3D shape classification, and it is very intuitive to think about 3D convolution(which requires huge computational power). The point cloud structure is very straightforward and friendly to sensor. Also, since we do not need an order of points, we can simply treat them as 1D array, and after some transformation, introduce a symmetric function to make the array unordered. Thus, 1D convolution made its sense.

I will continue working on data augmentation(such as shifting of data) and try to introduce intentionally flawed data(such as outliers and missing data, as the paper suggest that robustness of the model). 

I also noticed that the PointNet seems to vulnerable to rotation. I tried to add some random 3D rotations of object, and the performance drops quickly.

Also, as the project requires me to create my own database, I just write code to generate point clouds of some basic 3D shape. In the future, I may test the model on some online available dataset such as ModelNet10 and ModelNet40 if possible.



Some open question:

Symmetry, the most important key idea suggested in the paper, is implemented in the model by a simple global max pooling. I want to find some more interpretable ideas showing why such a simple symmetric function works so well.