import glob
import tensorflow as tf
from AlexNetSpec import AlexNetSpec as ANS   
from tensorflow import keras
from keras import layers
from keras.models import Sequential

batch_size = 32


train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
  "dataset",
  validation_split=0.2,
  subset="both",
  seed= 271,
  image_size=(ANS.HEIGHT, ANS.WIDTH),
  batch_size=batch_size)
  
class_names = train_ds.class_names

# Setup prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# making the model
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(ANS.HEIGHT, ANS.WIDTH, ANS.CHANNELS)),  # Normalization Layer
  layers.Conv2D(filters=96, kernel_size=(11,11), strides=4, activation='relu'), # 11x11 Kernel, Stride 4
  
  layers.MaxPooling2D(pool_size=(3,3),strides=2),                               # 3x3 MaxPool, Stride 2
  
  layers.ZeroPadding2D(padding=(2, 2)),                                         
  layers.Conv2D(filters=256, kernel_size=(5,5), activation='relu'),             # 5x5 Kernel, Pad 2

  layers.MaxPooling2D(pool_size=(3,3),strides=2),                               # 3x3 MaxPool, Stride 2
  
  layers.ZeroPadding2D(padding=(1, 1)),  
  layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu'),             # 3x3 Kernel, Pad 1

  layers.ZeroPadding2D(padding=(1, 1)),  
  layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu'),             # 3x3 Kernel, Pad 1

  layers.ZeroPadding2D(padding=(1, 1)),  
  layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'),             # 3x3 Kernel, Pad 1
  
  layers.MaxPooling2D(pool_size=(3,3),strides=2),                               # 3x3 MaxPool, Stride 2

  layers.Flatten(),                                                             # flatten 2D to 1D

  layers.Dense(4096, activation='relu'),                                        # 4096 Fully connected neurons
  layers.Dropout(0.5),                                                                           
  layers.Dense(4096, activation='relu'),                                        # 4096 Fully connected neurons
  layers.Dropout(0.5),
  layers.Dense(num_classes)                                                     # Classification Layer (classes)
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5
)

model.save('model')