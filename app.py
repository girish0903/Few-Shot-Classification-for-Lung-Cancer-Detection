from flask import Flask, render_template, request, jsonify, url_for
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
model_path = '10shotpn.pth'
convolutional_network = resnet12()
few_shot_classifier = PrototypicalNetworks(convolutional_network).to(DEVICE)
few_shot_classifier.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
few_shot_classifier.eval()

# Define transforms for input images
input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

sample_images_folder = 'static/samples'
sample_images = [img for img in os.listdir(sample_images_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = input_transform(image)
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
    try:
        # Placeholder for the support set (not used in this implementation)
        support_set_placeholder = None

        # Check the action from the submitted form
        action = request.form.get('action')

        image_path = None
        label = None
        label_class = None

        if action == 'sample':
            # If the "Submit Sample" button is clicked
            selected_sample = request.form.get('sample-images')
            image_path = os.path.join('static', 'samples', selected_sample)
            sample_image_url = '/' + image_path  # Ensure the leading slash
            result = {'image_path': sample_image_url, 'label': label, 'label_class': label_class, 'action': action}
        elif action == 'upload':
            # If the "Submit Upload" button is clicked
            uploaded_file = request.files['upload']
            if uploaded_file.filename != '':
                uploaded_filepath = os.path.join('static','temp_upload.jpg')
                uploaded_file.save(uploaded_filepath)
                image_path = uploaded_filepath
            else:
                return jsonify({'error': 'No file uploaded'})
        else:
            return jsonify({'error': 'Invalid action'})

        print(f"Action: {action}")
        print(f"Image Path: {image_path}")

        query_image = preprocess_image(image_path)

        with torch.no_grad():
            few_shot_classifier.eval()
            query_features = few_shot_classifier.compute_features(query_image)
            prototype_features = compute_prototype_features(support_set_placeholder)
            distances = cdist(query_features.cpu().numpy(), prototype_features.cpu().numpy(), 'euclidean')
            distances_tensor = torch.tensor(distances, device=DEVICE, dtype=torch.float32)
            query_predictions = -distances_tensor
            threshold = -14.675
            predicted_class = 0 if query_predictions.item() <= threshold else 1

        label = "Pneumonia" if predicted_class == 1 else "Normal"
        label_class = "pneumonia" if predicted_class == 1 else "normal"
        
        print(f"Query Predictions: {query_predictions.item()}")
        print(f"Threshold: {threshold}")


        if action == 'sample':
            result = {'image_path': image_path, 'label': label, 'label_class': label_class, 'action': action}
        else:
            result = {'image_path': 'temp_upload.jpg', 'label': label, 'label_class': label_class, 'action': action}
        print(f"Result: {result}")
        return render_template('index.html', result=result, sample_images=sample_images)
        # Pass the image path and other information to the HTML template

    except Exception as e:
        return jsonify({'error': str(e)})


def compute_prototype_features(support_set):
    if support_set is None:
        return torch.zeros((1, 640), device=DEVICE)
    return torch.zeros((len(support_set), 640), device=DEVICE)

if __name__ == '__main__':
    app.run(debug=True)
