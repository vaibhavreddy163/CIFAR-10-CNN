import tensorflow as tf
import pandas as pd

# 1. Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 2. Define the CNN architecture with variable depth
def build_cnn_model(depth=1):
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3)))
    
    # Add convolutional blocks based on depth
    for _ in range(depth):
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    # Flatten and fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

# 3. & 4. Train models of different depths and store the results
depths = [1, 2, 3, 4, 5]
results_df = pd.DataFrame(columns=['Depth', 'Training Accuracy', 'Validation Accuracy'])

for depth in depths:
    # Build and compile the model
    model = build_cnn_model(depth)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), verbose=0)
    
    # Get the final training and validation accuracies
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    # Append results to the DataFrame
    results_df = results_df.append({'Depth': depth, 
                                    'Training Accuracy': final_train_accuracy, 
                                    'Validation Accuracy': final_val_accuracy}, 
                                   ignore_index=True)

# 5. Display the results
print(results_df)
