## Face-Mask Detection

### Prerequisites
* Python 3.x
* Numpy 1.16.2
* OpenCV 4.5.1
* TensorFlow 1.13.1

*You can use **higher** version if you want*
*Tested on Windows 10 64bit*

### Folder Directory
* **dataset**: All images for training
* **face-mask-model.h5**: Trained model
* **haarcascade_frontalface_default.xml**: Haar Cascade for detecting faces
* **train_model.py**: used for training model
* **test_model**: used for testing (or running) the program

### Dataset
The dataset consists of 1376 images with 690 images containing images of people wearing masks and 686 images with people without masks. [Download](https://data-flair.training/blogs/download-face-mask-data/)

### How to run?
* Open cmd and change the path to folder includes **test_model.py** file
* Enter to commad line as below
    ```
    python test_model.py
    ```