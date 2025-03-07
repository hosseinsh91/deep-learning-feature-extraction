# Implementing Convolutional Filters on CIFAR-10

## **📌 Project Overview**
This project applies **convolutional filtering** on **CIFAR-10 images** using **TensorFlow/Keras** to demonstrate **feature extraction** in deep learning. It manually defines **3×3 convolutional filters** and applies them to images to observe the effect of convolution operations.

### **🚀 Key Features**
✅ **Loads and normalizes CIFAR-10 dataset**  
✅ **Manually defines 3×3 convolutional filters**  
✅ **Applies convolution using TensorFlow’s `tf.nn.conv2d` function**  
✅ **Visualizes original vs. filtered images**  

---

## **📌 Dataset: CIFAR-10**
- The **CIFAR-10 dataset** consists of **60,000 color images** categorized into:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Each image is **32x32 pixels with 3 color channels**.

### **📌 Data Preprocessing**
- The dataset is **normalized** by scaling pixel values to `[0,1]`:
  ```python
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
