# Neural-Networks-for-Handwritten-Digit-Recognition-Multiclass
This repository contains code for building and training neural networks to recognize handwritten digits from the MNIST dataset.

Handwritten digit recognition is a classic problem in the field of machine learning and computer vision. It involves identifying the digits present in images of handwritten characters. 

Requirements

To run the code in this repository, you'll need the following dependencies:

Python (>= 3.6)
NumPy
TensorFlow (>= 2.0)
Matplotlib (for visualization, optional)

You can install the required packages via pip:

pip install numpy tensorflow matplotlib

I have used the following code for the softmax function.

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    
    
    denominator = np.sum(np.exp(z))

    # Compute the softmax function
    a = np.exp(z) / denominator
    
    return a
Contact
For any questions or inquiries, please contact deep200patel@gmail.com.
    
