import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

def download_image(image_url):
    response = requests.get(image_url)
    response.raise_for_status()  # Check if the download was successful
    return Image.open(BytesIO(response.content))

def classify_image(image_url, model_path, img_height, img_width, class_names):
    model = load_model(model_path)

    # Download and preprocess the image
    img = download_image(image_url)
    img = img.resize((img_width, img_height))
    img_array = keras_image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

# Get the image URL from the user
img_url = input("Enter the URL of the image: ")
model_path = 'scenary_classification_model'
img_height = 150  # Replace with the height used during training
img_width = 150   # Replace with the width used during training
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # Replace with your class names

predicted_class, confidence = classify_image(img_url, model_path, img_height, img_width, class_names)
print(f"This image most likely belongs to {predicted_class} with a {confidence:.2f} percent confidence.")
