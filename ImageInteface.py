import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Define the same transformations as used during training
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Model definition (same as your training code)
class FaceRecognitionCNN(torch.nn.Module):
    def __init__(self, num_classes=40):
        super(FaceRecognitionCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 23 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    st.title('Face Recognition Prediction by Balan Nagarajan')

    uploaded_file = st.file_uploader('Upload an image file (.pgm)', type=['pgm'])

    # Your mapping
    mapping = {
        's1': 0, 's10': 1, 's11': 2, 's12': 3, 's13': 4, 's14': 5, 's15': 6, 's16': 7, 's17': 8, 's18': 9,
        's19': 10, 's2': 11, 's20': 12, 's21': 13, 's22': 14, 's23': 15, 's24': 16, 's25': 17, 's26': 18,
        's27': 19, 's28': 20, 's29': 21, 's3': 22, 's30': 23, 's31': 24, 's32': 25, 's33': 26, 's34': 27,
        's35': 28, 's36': 29, 's37': 30, 's38': 31, 's39': 32, 's4': 33, 's40': 34, 's5': 35, 's6': 36,
        's7': 37, 's8': 38, 's9': 39
    }

    st.sidebar.title("References folder to class label. For example if you pick an image from s2 folder, class label is 11")

    # Show each key-value pair
    for key, value in mapping.items():
        st.sidebar.write(f"{key}: {value}")

    if uploaded_file is not None:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FaceRecognitionCNN(num_classes=40)
        model.load_state_dict(torch.load('face_recognition_model.pth', map_location=device))
        model.to(device)
        model.eval()

        # Read image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Transform image
        input_tensor = data_transforms(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()

        #st.write(f'Predicted class: {pred_class}')
        st.markdown(
            f'<p style="font-size:32px; font-weight:bold; color:red;">Predicted class: {pred_class}</p>',
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()
