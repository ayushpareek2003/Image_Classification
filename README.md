# IMage_classification

Trained a Machine Learning Model to recognize Image . Used CNN layers 
first we imported basic libraries like NumPy  after that for training and processing we imported cv2 ,TensorFlow (we used keras for training) , and for visualization Matplotlib and Seaborn (you will not see that much of visualisation in our code because when we pasted our code from jupyter notebook to python file we remove those lines) . after this we imported sklearn for shuffling  our dataset.

2.  The variable name (class_names) defines a list of class names (labels mentioned in boilerplate code), and c_n_label is a dictionary that maps class names to integer labels. These labels are used for training and evaluation.

3. The load_d function is defined to load the training and testing image data from specified directories. It reads images from folders corresponding to the class labels, converts them to the RGB color format, resizes them to a common size of 150x150 pixels, and stores them in NumPy arrays. The function returns two tuples, one for training data and labels (T_i and T_l) and another for testing data and labels (t_i and t_l). Data shuffling is also performed for the training set.

4. we created a dis function  to display a grid of example images from the dataset, along with their corresponding class labels.

5. The images in the training and testing sets are normalized by dividing them by 255.0. This step scales pixel values to the range [0, 1].

6. A convolutional neural network model is defined using TensorFlow's Keras . It's a simple CNN with two convolutional layers, max-pooling layers, and fully connected layers. The model architecture is as follows:
   Convolutional Layer (32 filters, 3x3 kernel, ReLU activation)
   Max Pooling Layer (2x2 pool size)
   Convolutional Layer (32 filters, 3x3 kernel, ReLU activation)
   Max Pooling Layer (2x2 pool size)
   Flattening Layer
   Fully Connected Layer (128 units, ReLU activation)
   Fully Connected Layer (50 units, Softmax activation)
   Model Compilation: The model is compiled with the Adam optimizer, sparse categorical cross-entropy, loss function.

8. The model is trained on the training data (T_i (Images), T_l (labels)) using the fit method. It specifies a batch size of 128, 15 epochs, and a 20% validation split.

9. After training  the model's  is saved as a file named "Trained_Weight.h5"
 
10. we loaded our model in boilerplate code and  the image which come as input we resizes it to a target size of 150x150 pixels. Converted the image to a NumPy array.
     after this Added an extra dimension to the array, making it suitable for batch processing. after this we Preprocessed the image according to the requirements of the ResNet-50 model.   
This step  involves mean subtraction and scaling to match the model's pretraining.


Make Predictions with the Model:

11. output from model will come in a array ,which is an array of class probabilities. probablity at any index represents the probablity of belonging of this image to the label present in label array at the same index.

12. we have a list of class names that correspond to the output classes of the model. after this we use (argmax) from numpy library on output array to find the index of the class with the highest probability in the predictions array, we retrieves the class name associated with the predicted index. This will be our final answer.
