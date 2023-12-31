from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
from easyfsl.methods import PrototypicalNetworks
from easyfsl.modules import resnet12
import os
from scipy.spatial.distance import cdist

app = Flask(__name__)

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_path = 'model.pth'
convolutional_network = resnet12()
few_shot_classifier = PrototypicalNetworks(convolutional_network).to(DEVICE)
few_shot_classifier.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
few_shot_classifier.eval()

# Define transforms for input images
input_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

sample_images_folder = 'static/samples'
sample_images = [img for img in os.listdir(sample_images_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # Apply the same transform used in the dataset
    image = input_transform(image)

    # Ensure the image has a batch dimension
    image = image.unsqueeze(0)

    return image.to(DEVICE)

@app.route('/')
def home():
    return render_template('index.html', sample_images=sample_images)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    selected_sample = request.form.get('sample-images')
    image_path = os.path.join(sample_images_folder, selected_sample)

    try:
        # Placeholder for the support set (not used in this implementation)
        support_set_placeholder = None

        # Preprocess the query image
        query_image = preprocess_image(image_path)

        # Make predictions for the single image
        with torch.no_grad():
            few_shot_classifier.eval()

            # Extract query features
            query_features = few_shot_classifier.compute_features(query_image)

            # Compute prototype features from the support set (replace None with actual prototype features)
            prototype_features = compute_prototype_features(support_set_placeholder)

            # Compute pairwise distances between query and prototype features
            distances = cdist(query_features.cpu().numpy(), prototype_features.cpu().numpy(), 'euclidean')

            # Convert distances to PyTorch tensor
            distances_tensor = torch.tensor(distances, device=DEVICE, dtype=torch.float32)

            # Use negative distances as scores (assuming you want to minimize distances)
            query_predictions = -distances_tensor
            print("Raw Scores (before softmax):", query_predictions)

            # Apply the threshold for binary classification
            threshold = -13.85
            predicted_class = 0 if query_predictions.item() <= threshold else 1

        # Define class labels
        label = "Cancerous" if predicted_class == 1 else "Normal"
        label_class = "cancerous" if predicted_class == 1 else "normal"

        # Return the predicted class and label in the result
        result = {'image_path': image_path, 'label': label, 'label_class': label_class}
        return render_template('index.html', result=result, sample_images=sample_images)

    except Exception as e:
        return jsonify({'error': str(e)})

def compute_prototype_features(support_set):
    if support_set is None:
        # Return a default tensor when the support set is None
        return torch.zeros((1, 640), device=DEVICE)

    # Replace this with the actual computation of prototype features from the support set
    # Make sure to return a torch.Tensor
    return torch.zeros((len(support_set), 640), device=DEVICE)  # Example placeholder

if __name__ == '__main__':
    app.run(debug=True)
