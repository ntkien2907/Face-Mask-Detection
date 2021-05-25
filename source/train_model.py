# Import all libraries and modules required
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense


# Build the neural network
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3,3),activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())


# Image Data Augmentation
TRAINING_DIR = "dataset/train"

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=16,
    target_size=(150,150),
)

VALIDATION_DIR = "dataset/test"

validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=16,
    target_size=(150,150),
)


# Train model
model_saved = model.fit_generator(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
)


# Save model
model.save('face-mask-model.h5', model_saved)