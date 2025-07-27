import tensorflow as tf 
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
# Create the CNN model
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # Reshape to add a channel dimension
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # Output layer with 10 units for 10 classes (0-9)
])
# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=32)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')
# Load and preprocess the new image
new_image = Image.open(r'5.jpg').convert('L')
new_image = new_image.resize((28, 28))
new_image = np.array(new_image) / 255.0
new_image = np.expand_dims(new_image, axis=0)

# Classify the new image using the trained model
predictions = model.predict(new_image)
predicted_label = np.argmax(predictions[0])

print(f"Prediction for new image: {predicted_label}")