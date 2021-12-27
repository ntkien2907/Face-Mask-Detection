import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from utils import save_figures

input_shape = (224, 224, 3)
target_size = (224, 224)
batch_size = 32
n_episodes = 20
learning_rate = 1e-4

data_dir = 'data/'
image_paths = list(paths.list_images(data_dir))
data, labels = [], []

# Loop over the dataset
for image_path in image_paths:
    # Extract label from the filename
    label = image_path.split(os.path.sep)[-2]
    # Load the input image and preprocess
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    # Update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# Convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

# One-hot encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded)

# Class name
target_names = list(map(lambda x: x.replace(data_dir, ''), set(labels)))

# Split dataset into 3 parts: train (60%), validation (20%) and test (20%)
trainX, testX, trainY, testY = train_test_split(data, onehot_encoded_labels, test_size=0.4, stratify=onehot_encoded_labels, random_state=1)
valX, testX, valY, testY = train_test_split(testX, testY, test_size=0.5, stratify=testY, random_state=1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
)

# Load the MobileNetV2 network without output layer
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))

# Our custom model
custom_model = base_model.output
custom_model = MaxPooling2D(pool_size=(7,7))(custom_model)
custom_model = Flatten()(custom_model)
custom_model = Dense(128, activation="relu")(custom_model)
custom_model = Dropout(0.5)(custom_model)
custom_model = Dense(3, activation="softmax")(custom_model)

# Concatenate MobileNetV2 to our custom model
model = Model(inputs=base_model.input, outputs=custom_model)

# Mark loaded layers as not trainable
for layer in base_model.layers:
    layer.trainable = False

# Compile model
opt = Adam(learning_rate=learning_rate, decay=learning_rate/n_episodes)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

# Train model
print('\n[INFO] training network ...')
history = model.fit(
    datagen.flow(trainX, trainY, batch_size=batch_size), 
    steps_per_epoch=np.ceil(len(trainX)/batch_size), 
    epochs=n_episodes, 
    validation_data=(valX, valY), 
    validation_steps=np.ceil(len(valX)/batch_size),
)

# Save history for accuracy and loss
save_figures(history, dir='figures/')

# Evaluate model
print('\n[INFO] evaluating network ...')
y_preds = model.predict(testX, batch_size=batch_size, steps=np.ceil(len(testX)/batch_size))
y_preds = np.argmax(y_preds, axis=1)
print(classification_report(testY.argmax(axis=1), y_preds, target_names=target_names))

# Save model
print('\n[INFO] saving model ...')
model.save('face-mask-model.h5')
