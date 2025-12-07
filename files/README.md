# TensorFlow Image Classifier - Streamlit App

A simple Streamlit web application that loads a TensorFlow model (.h5 file) and allows users to upload images for classification.

## Features

- 📁 Load any TensorFlow model from an .h5 file
- 🖼️ Upload images (JPG, PNG, BMP formats)
- 🔍 Automatic image preprocessing (resize, normalize)
- 📊 Display prediction results with confidence scores
- 🏗️ View model architecture
- ⚙️ Configurable image input size

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. In the sidebar, enter the path to your TensorFlow model (.h5 file)
   - Default path is `model.h5` in the current directory
   - You can use absolute or relative paths

3. Configure the image preprocessing settings:
   - Set the image width and height to match your model's expected input
   - Default is 224x224 (common for many pre-trained models)

4. Upload an image using the file uploader

5. View the prediction results!

## Model Requirements

Your TensorFlow model should:
- Be saved in .h5 format (Keras model format)
- Accept image inputs with shape `(batch_size, height, width, channels)`
- Have been trained with normalized pixel values (0-1 range)

## Image Preprocessing

The app automatically:
- Resizes images to the specified dimensions
- Converts grayscale images to RGB (if needed)
- Removes alpha channel (if present)
- Normalizes pixel values to 0-1 range
- Adds batch dimension

## Customization

If your model requires different preprocessing:

1. Modify the `preprocess_image()` function in `app.py`
2. Adjust normalization (e.g., for models trained with ImageNet preprocessing)
3. Change the target size to match your model's input shape

## Example Models

This app works with various TensorFlow/Keras models:
- Custom trained models
- Pre-trained models (VGG, ResNet, MobileNet, etc.)
- Transfer learning models

## Troubleshooting

**Model won't load:**
- Check the file path is correct
- Ensure the .h5 file is a valid Keras model
- Check file permissions

**Prediction errors:**
- Verify image size matches model input
- Check if model expects different normalization
- Ensure image format is compatible

**Performance issues:**
- Large models may take time to load (cached after first load)
- High-resolution images may slow predictions

## Notes

- The model is cached using `@st.cache_resource` for better performance
- Supports both binary and multi-class classification
- Automatically handles different image formats and color modes
