# Implementation-of-Neural-Network
Implementing Neural Network on Irish dataset 

On Irish dataset (uploaded on Irish folder here), this algorithm evalutes: accuracy, Mean Squared Error (MSE), Crossentropy and log-likelihood. Followed by that, the function returns those five results that are correctcount, accuracy, MSE, Crossentropy and Loglikelihood in a list.


This code is adopted fro NNDL book by Nielson from Chapter 1; the original code is network.py (uploaded here on the the dataset folder.) In this code the target variable (y) of a dataset is assumed to be a vector of the form 'one-hot-vector' representation, which is a list of all 0's with exactly 1 for the target class. For instance, if there were four target classes in the dataset aka multiclass classification problem, and a specific instance's target class was 'three', the target was encoded as [0,0,1,0]. This encoding scheme is known as 'categorical' format. 


<p align="center">
<img src = "images/IRISH.png">
 </p>

On Nielson's code, this script (which is NN_network.py) edits the evakute() function. MSE from Eq. (6), cross-entropy from Eq. (57 or 63) and loglikelihood from Eq. (80). NOTE: on loglikelihood equation 80, the formula is missing 1/n in the begining on Nielson's book. 

NOTE: Each cost function must return a scalar value, not an array. 

- For MSE and Cross-entropy look at the two function classes, QuadraticCost and CrossEntropyCost on network2.py file.

- For loglikelihood, you have to pick out the activation value of a node for which the target y array has a one - represented by (binarised) 'one-hot-vector'. Provided that you get first obtain the index to the node by calling argmax to the target y and give the index to the output layer's activation array, one would return a probelm -- Numpy's subcript operator returns an array with one element, instead of a scalar. This is because the activation values of a layer are stored in column vectors rather than row. 


<img src = "images/row_column.png" width = "480" height = "310">

The goal is to generate the following curves of Test versus Train:

<img src = "images/error_cruves.png" width = "480" height = "310">

<img src = "images/test_v_train.png" width = "480" height = "310">

<img src = "images/test_v_train_1.png" width = "480" height = "310">

<img src = "images/test_v_train_2.png" width = "480" height = "310">




From "Neural Network and Deep Learning" book by Michael Nielsen. Free online version available @ http://neuralnetworksanddeeplearning.com/


