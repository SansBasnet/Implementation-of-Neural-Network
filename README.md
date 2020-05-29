# Implementation-of-Neural-Network
Implementing Neural Network on Irish dataset 

On Irish dataset (uploaded on Irish folder here), this algorithm evalutes: accuracy, Mean Squared Error (MSE), Crossentropy and log-likelihood. Followed by that, the function returns those five results that are correctcount, accuracy, MSE, Crossentropy and Loglikelihood in a list.


This code is adopted fro NNDL book by Nielson from Chapter 1; the original code is network.py (uploaded here on the the dataset folder.) In this code the target variable (y) of a dataset is assumed to be a vector of the form 'one-hot-vector' representation, which is a list of all 0's with exactly 1 for the target class. For instance, if there were four target classes in the dataset aka multiclass classification problem, and a specific instance's target class was 'three', the target was encoded as [0,0,1,0]. This encoding scheme is known as 'categorical' format. 

<br>
<p align="center">
<img src = "images/IRISH.png">
 </p>
<br>

On Nielson's code, this script (which is NN_network.py) edits the evakute() function. MSE from Eq. (6), cross-entropy from Eq. (57 or 63) and loglikelihood from Eq. (80). NOTE: on loglikelihood equation 80, the formula is missing 1/n in the begining on Nielson's book. 

NOTE: Each cost function must return a scalar value, not an array. 

- For MSE and Cross-entropy look at the two function classes, QuadraticCost and CrossEntropyCost on network2.py file.

- For loglikelihood, you have to pick out the activation value of a node for which the target y array has a one - represented by (binarised) 'one-hot-vector'. Provided that you get first obtain the index to the node by calling argmax to the target y and give the index to the output layer's activation array, one would return a probelm -- Numpy's subcript operator returns an array with one element, instead of a scalar. This is because the activation values of a layer are stored in column vectors rather than row. Check out the code below for that: 

<p align="center">
<img src = "images/row_column.png">
 </p>

We are also editing the functions SGD() for training_data, at the end of each epoch, and print the resturned results. 
Collecting the performance results from evaluate() for all epochs for trainin_data and test_data into indovidual lists, and return the two lists in a list. 

First split the dataset randomly into 70% training and 30% test. This script does not call SciKitlearn or other packages but rather is about hard coding. It beings with shuffling the instances in the original dataset, and takes the first 70% as the training and the rest as the test. Then create a new network with randomly initialized weights of the size [4,20,3].  Create a new network by simply calling the constructor as: net4 = network.Network([4,20,3]). Then train the network for 50 epochs with eta = 0.1 and the mini batch size = 5 (and take the default for stopaccuracy).  Save the results.

Test versus Train output should look something like this:


<img src = "images/error_cruves.png" width = "480" height = "310">
<img src = "images/test_v_train.png" width = "480" height = "310">

<img src = "images/test_v_train_1.png" width = "480" height = "310">

<img src = "images/test_v_train_2.png" width = "480" height = "310">



From "Neural Network and Deep Learning" book by Michael Nielsen. Free online version available @ http://neuralnetworksanddeeplearning.com/


