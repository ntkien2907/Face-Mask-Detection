## Face-Mask Detection

![Face-Mask Detector](./images/demo.gif)

### Prerequisites
* Python 3.6
* Tensorflow 2.5.0
* Keras 2.6.0

*Tested on Windows 10 - 64 bit*

---
### Source folder
* **data**: All preprocessed images (128x128) for training, validating and testing. You can download the original images following to the next section.
* **figures**: Contain accuracy, loss and classification report.
* **face-mask-model.h5**: Trained model.
* **preprocess.py**: Process the dataset.
* **test.py** : Run the program via webcam.
* **train.py**: Train and evaluate model.
* **utils.py**: A collection of small functions which make common patterns shorter and easier.

---
### Dataset
* The used dataset is aggregated from three datasets:
    * Correctly Masked Face Dataset ([CMFD](https://github.com/cabani/MaskedFace-Net))
    * Incorrectly Masked Face Dataset ([IMFD](https://github.com/cabani/MaskedFace-Net))
    * Flickr-Faces-HQ Dataset ([FFHQ](https://github.com/NVlabs/ffhq-dataset))
* There are 3 classes: [incorrectly-mask](https://esigelec-my.sharepoint.com/personal/cabani_esigelec_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcabani%5Fesigelec%5Ffr%2FDocuments%2FMaskedFaceNetDataset%2FIMFD&originalPath=aHR0cHM6Ly9lc2lnZWxlYy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9jYWJhbmlfZXNpZ2VsZWNfZnIvRWlyalM4ZXc3LTVMbk84STU2VWs2M3dCS2Vid1NsdWtGQkZCYU84TjI1d24zZz9ydGltZT1NUThJd3JBbDJVZw), [with-mask](https://esigelec-my.sharepoint.com/personal/cabani_esigelec_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcabani%5Fesigelec%5Ffr%2FDocuments%2FMaskedFaceNetDataset%2FCMFD&originalPath=aHR0cHM6Ly9lc2lnZWxlYy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9jYWJhbmlfZXNpZ2VsZWNfZnIvRXYzR2RuUVN5enhQanl6VTVFbEhxYWdCbGtSQ2FLbm5DSTg1aVgtZDFMNE9IQT9ydGltZT1faHo2MXJBbDJVZw) and [without-mask](https://drive.google.com/drive/folders/1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv).
* Download from folder **00000** to folder **11000**. If you want a larger dataset, please visit the above links to download more.

---
### How to use
* Step 1: Preprocessing the dataset which you have already dowloaded manually. *Skip this step if using the attached dataset*.
    ```
    python preprocess.py
    ```
* Step 2: Training model. *Skip this step if using the attached model*.
    ```
    python train.py
    ```
* Step 3: Testing model through your webcam.
    ```
    python test.py
    ```

* **Notice:** This model works best with the mask below.
<p align='middle'><img src='./images/face-mask.jpg' width=25% /></p>

---
### Evaluation

<p align='middle'>
  <img src='./source/figures/accuracy.jpg' width=48% />
  <img src='./source/figures/loss.jpg' width=48% /> 
</p>
<p align='middle'><img src='./source/figures/evaluation.jpg' width=50% /></p>

---
### References
1. [Adrian Rosebrock (2020). COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning.](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
2. [Hussain Mujtaba (2021). Real-time Face detection | Face Mask Detection using OpenCV.](https://www.mygreatlearning.com/blog/real-time-face-detection/)
3. [Ryan Akilos (2017). A simple example: Confusion Matrix with Keras flow_from_directory.py](https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045)
4. [Chandrika Deb. Face-Mask-Detection. Github.](https://github.com/chandrikadeb7/Face-Mask-Detection)
5. [Real-Time Face Mask Detector with Python, OpenCV, Keras.](https://data-flair.training/blogs/face-mask-detection-with-python/)
