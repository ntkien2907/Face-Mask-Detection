# Import all libraries and modules required
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import Callback


# Callback function
class CustomCallbacks(Callback):
    def on_epoch_end(self, epoch, logs={}):
      if(logs.get('acc') > 0.95):
        print("\n95% accuracy reached")
        self.model.stop_training = True


# Build the neural network
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())


# Image Data Augmentation
batch_size = 30

TRAINING_DIR = "face-mask-dataset-3classes/train"
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(128,128),
    batch_size=batch_size,
    class_mode='categorical',
)

VALIDATION_DIR = "face-mask-dataset-3classes/test"
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(128,128),
    batch_size=batch_size,
    class_mode='categorical',
)

# Train model
model_saved = model.fit_generator(
    train_generator,
    steps_per_epoch=2000//batch_size,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[CustomCallbacks()],
)

# Save model
model.save('face-mask-model.h5', model_saved)