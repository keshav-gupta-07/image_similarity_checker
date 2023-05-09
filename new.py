import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np


# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Remove the fully connected layer
model = nn.Sequential(*list(model.children())[:-1])

# Set model to evaluation mode
model.eval()

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((1024, 1440)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB');
    img = img.resize((1440, 1024))
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor
# Function to compute similarity score between two images
def image_similarity(image1_path, image2_path):
    # Preprocess images
    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    # Get features from pre-trained model
    with torch.no_grad():
        features1 = model(img1)
        features2 = model(img2)

    # Flatten features and compute cosine similarity
    features1 = features1.flatten().numpy()
    features2 = features2.flatten().numpy()
    similarity_score = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    
    return similarity_score

# Example usage
similarity_percentage = image_similarity('test1.png', 'test2.png')
print('Similarity percentage:', similarity_percentage*100)