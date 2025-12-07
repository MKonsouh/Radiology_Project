"""
Example script to create a simple TensorFlow model for testing the Streamlit app
This creates a basic image classifier that can be used to test the app functionality
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_simple_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Create a simple CNN model for image classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_binary_model(input_shape=(224, 224, 3)):
    """
    Create a simple binary classification model
    
    Args:
        input_shape: Shape of input images (height, width, channels)
    
    Returns:
        Compiled Keras model for binary classification
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("Creating example TensorFlow models...")
    
    # Create a multi-class classification model
    print("\n1. Creating multi-class model (10 classes)...")
    multiclass_model = create_simple_model(num_classes=10)
    multiclass_model.save('model_multiclass.h5')
    print("   ✓ Saved as 'model_multiclass.h5'")
    
    # Create a binary classification model
    print("\n2. Creating binary classification model...")
    binary_model = create_binary_model()
    binary_model.save('model_binary.h5')
    print("   ✓ Saved as 'model_binary.h5'")
    
    # Create a smaller model for faster testing
    print("\n3. Creating small test model...")
    small_model = create_simple_model(input_shape=(128, 128, 3), num_classes=5)
    small_model.save('model_small.h5')
    print("   ✓ Saved as 'model_small.h5'")
    
    print("\n" + "="*50)
    print("Models created successfully!")
    print("="*50)
    print("\nYou can now use these models with the Streamlit app:")
    print("  streamlit run app.py")
    print("\nThen enter one of these paths in the sidebar:")
    print("  - model_multiclass.h5")
    print("  - model_binary.h5")
    print("  - model_small.h5")
