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
    st.set_page_config(page_title="Face Recognition by Balan Nagarajan", layout="wide")

    # Title
    st.markdown(
        """
        <h1 style="color: #2c3e50; text-align:center; margin-bottom: 10px;">
            🧑‍💻 Face Recognition Prediction by Balan Nagarajan
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Fancy info sections using columns and custom styling
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style="padding: 18px; background-color: #f0f2f6; border-radius: 12px; margin: 10px 0;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">📚 Dataset Reference</h3>
                <p style="font-size: 15px;">
                    For training images and class details:<br>
                    <a href="https://www.kaggle.com/datasets/kasikrit/att-database-of-faces" style="color: #3498db; text-decoration: none;" target="_blank">
                        👉 Kaggle ATT Faces Dataset
                    </a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div style="padding: 18px; background-color: #f0f2f6; border-radius: 12px; margin: 10px 0;">
                <h3 style="color: #2c3e50; margin-bottom: 8px;">🖼️ Testing Tips</h3>
                <p style="font-size: 15px;">
                    Use <span style="color: #e74c3c; font-weight: bold;">.pgm images</span> from the dataset.<br>
                    Sample images available on Kaggle.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    uploaded_file = st.file_uploader('Upload an image file (.pgm)', type=['pgm'])

    # Enhanced sidebar with card-like styling
    st.sidebar.markdown(
        """
        <div style="padding: 18px; background-color: #2c3e50; border-radius: 12px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0; text-align:center;">🧑🏫 Class Reference Guide</h2>
            <p style="color: #ecf0f1; font-size: 13px; text-align:center;">
                If you pick an image from <b>s2</b> folder, class label is <b>11</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Your mapping
    mapping = {
        's1': 0, 's10': 1, 's11': 2, 's12': 3, 's13': 4, 's14': 5, 's15': 6, 's16': 7, 's17': 8, 's18': 9,
        's19': 10, 's2': 11, 's20': 12, 's21': 13, 's22': 14, 's23': 15, 's24': 16, 's25': 17, 's26': 18,
        's27': 19, 's28': 20, 's29': 21, 's3': 22, 's30': 23, 's31': 24, 's32': 25, 's33': 26, 's34': 27,
        's35': 28, 's36': 29, 's37': 30, 's38': 31, 's39': 32, 's4': 33, 's40': 34, 's5': 35, 's6': 36,
        's7': 37, 's8': 38, 's9': 39
    }

    # Display mapping in styled boxes
    for key, value in mapping.items():
        st.sidebar.markdown(
            f"""
            <div style="padding: 8px 12px; background-color: #f8f9fa; border-radius: 8px; margin: 4px 0; 
                        border-left: 5px solid #3498db; display: flex; justify-content: space-between;">
                <span style="font-weight: 600; color: #2c3e50;">{key}</span>
                <span style="color: #e74c3c; font-weight: bold;">{value}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if uploaded_file is not None:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FaceRecognitionCNN(num_classes=40)
        model.load_state_dict(torch.load('face_recognition_model.pth', map_location=device))
        model.to(device)
        model.eval()

        # Read image
        image = Image.open(uploaded_file)
        # Let the user pick the width
        width = st.slider("Select image width (pixels)", min_value=100, max_value=1000, value=400)
        st.image(image, caption='Uploaded Image', width=width)

        # Transform image
        input_tensor = data_transforms(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()

        st.markdown(
            f'''
            <div style="text-align:center; margin-top: 30px;">
                <span style="font-size:36px; font-weight:bold; color:#e74c3c;">
                    Predicted class: {pred_class}
                </span>
            </div>
            ''',
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()
