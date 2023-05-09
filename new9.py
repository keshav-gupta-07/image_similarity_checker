import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Load pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to preprocess image for ResNet50 model
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Function to compute similarity score between two images
def image_similarity(image1_path, image2_path):
    # Preprocess images
    img1 = preprocess(image1_path)
    img2 = preprocess(image2_path)

    # Get features from pre-trained model
    features1 = model.predict(img1)
    features2 = model.predict(img2)

    # Flatten features and compute cosine similarity
    features1 = features1.flatten()
    features2 = features2.flatten()
    similarity_score = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    
    return similarity_score

# Example usage
similarity_percentage = image_similarity('hii2.png', 'test2.png')
print('Similarity percentage:', similarity_percentage)

tf.saved_model.save(model, './saved_model')