## Face-Mask Detection

![Face-Mask Detector](./images/demo.gif)

### Prerequisites
* Python 3.x
* OpenCV 4.5.1
* TensorFlow 2.1.0
* Keras 2.3.1

*You can use **higher** version if you want*

*Tested on Windows 10 64bit*

### Folder Directory
* **face-mask-dataset-3classes**: All images for training and testing. I do not push here because it is quite large (~5.71GB). You have to create and download by yourself following to the instruction below (in **Dataset** section) if you want to train model from scratch.
* **face-mask-model.h5**: Trained model
* **train_model.py**: Used for training model
* **test_model**: Used for running the program

### Dataset
* The used dataset is aggregated from three datasets:
    * Correctly Masked Face Dataset ([CMFD](https://github.com/cabani/MaskedFace-Net))
    * Incorrectly Masked Face Dataset ([IMFD](https://github.com/cabani/MaskedFace-Net))
    * Flickr-Faces-HQ Dataset ([FFHQ](https://github.com/NVlabs/ffhq-dataset))
* There are 3 classes: [incorrectly-mask](https://esigelec-my.sharepoint.com/personal/cabani_esigelec_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcabani%5Fesigelec%5Ffr%2FDocuments%2FMaskedFaceNetDataset%2FIMFD&originalPath=aHR0cHM6Ly9lc2lnZWxlYy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9jYWJhbmlfZXNpZ2VsZWNfZnIvRWlyalM4ZXc3LTVMbk84STU2VWs2M3dCS2Vid1NsdWtGQkZCYU84TjI1d24zZz9ydGltZT1NUThJd3JBbDJVZw), [with-mask](https://esigelec-my.sharepoint.com/personal/cabani_esigelec_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcabani%5Fesigelec%5Ffr%2FDocuments%2FMaskedFaceNetDataset%2FCMFD&originalPath=aHR0cHM6Ly9lc2lnZWxlYy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9jYWJhbmlfZXNpZ2VsZWNfZnIvRXYzR2RuUVN5enhQanl6VTVFbEhxYWdCbGtSQ2FLbm5DSTg1aVgtZDFMNE9IQT9ydGltZT1faHo2MXJBbDJVZw), [without-mask](https://drive.google.com/drive/folders/1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv)
* Download from **00000** folder to **09000** folder
* Here is the structure of the dataset

	![Dataset Directory](./images/dataset-directory.png)

### How to use
* Open cmd and change the path to folder which contains **test_model.py** file
* Enter to command line as below
    ```
    python test_model.py
    ```

### References
1. [Adrian Rosebrock (2020). COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
2. [Hussain Mujtaba (2021). Real-time Face detection | Face Mask Detection using OpenCV](https://www.mygreatlearning.com/blog/real-time-face-detection/)
3. [Real-Time Face Mask Detector with Python, OpenCV, Keras](https://data-flair.training/blogs/face-mask-detection-with-python/)