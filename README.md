🧠 CIFAR-10 Image Classification using CNN

This project demonstrates an end-to-end Convolutional Neural Network (CNN) model built with TensorFlow/Keras to classify images from the CIFAR-10 dataset into 10 distinct categories.
It also includes a Streamlit web app for real-time image prediction.

📁 Dataset

The CIFAR-10 dataset consists of 60,000 color images (32×32 pixels) in 10 classes, with 6,000 images per class:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The dataset was split as follows:

Training: 70%

Validation: 15%

Testing: 15%

🏗️ Model Architecture

The CNN model includes:

6 Convolutional layers with Batch Normalization and ReLU activation

MaxPooling and Dropout for regularization

Dense layers with softmax activation for classification

Optimizer: Adam
Loss Function: Categorical Crossentropy
Callbacks: EarlyStopping, ReduceLROnPlateau

📊 Performance

Validation Accuracy: ~0.86

Test Accuracy: ~0.87

F1-Score: 0.87 (macro average)

The model generalizes well with minimal overfitting, as shown by training vs. validation loss and accuracy curves.

🖼️ Results

Visualized confusion matrix and classification report for all 10 classes
🧠 CIFAR-10 Image Classification using CNN

This project demonstrates an end-to-end Convolutional Neural Network (CNN) model built with TensorFlow/Keras to classify images from the CIFAR-10 dataset into 10 distinct categories.
It also includes a Streamlit web app for real-time image prediction.

📁 Dataset

The CIFAR-10 dataset consists of 60,000 color images (32×32 pixels) in 10 classes, with 6,000 images per class:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The dataset was split as follows:

Training: 70%

Validation: 15%

Testing: 15%

🏗️ Model Architecture

The CNN model includes:

6 Convolutional layers with Batch Normalization and ReLU activation

MaxPooling and Dropout for regularization

Dense layers with softmax activation for classification

Optimizer: Adam
Loss Function: Categorical Crossentropy
Callbacks: EarlyStopping, ReduceLROnPlateau

📊 Performance

Validation Accuracy: ~0.86

Test Accuracy: ~0.87

F1-Score: 0.87 (macro average)

The model generalizes well with minimal overfitting, as shown by training vs. validation loss and accuracy curves.

🖼️ Results

Visualized confusion matrix and classification report for all 10 classes
🧠 CIFAR-10 Image Classification using CNN

This project demonstrates an end-to-end Convolutional Neural Network (CNN) model built with TensorFlow/Keras to classify images from the CIFAR-10 dataset into 10 distinct categories.
It also includes a Streamlit web app for real-time image prediction.

📁 Dataset

The CIFAR-10 dataset consists of 60,000 color images (32×32 pixels) in 10 classes, with 6,000 images per class:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The dataset was split as follows:

Training: 70%

Validation: 15%

Testing: 15%

🏗️ Model Architecture

The CNN model includes:

6 Convolutional layers with Batch Normalization and ReLU activation

MaxPooling and Dropout for regularization

Dense layers with softmax activation for classification

Optimizer: Adam
Loss Function: Categorical Crossentropy
Callbacks: EarlyStopping, ReduceLROnPlateau

📊 Performance

Validation Accuracy: ~0.86

Test Accuracy: ~0.87

F1-Score: 0.87 (macro average)

The model generalizes well with minimal overfitting, as shown by training vs. validation loss and accuracy curves.

🖼️ Results

Visualized confusion matrix and classification report for all 10 classes
<img width="1162" height="830" alt="image" src="https://github.com/user-attachments/assets/7fc60790-cee4-401b-8318-4686f7111a7b" />
<img width="1328" height="846" alt="image" src="https://github.com/user-attachments/assets/d31749de-b488-4535-a1d0-a3567ebe0881" />
<img width="1324" height="970" alt="image" src="https://github.com/user-attachments/assets/928fa9c1-fca1-4552-ae6e-5923f6c0acef" />
<img width="1444" height="768" alt="image" src="https://github.com/user-attachments/assets/eadc7e8d-be21-4465-a09f-33d2732ce16d" />
<img width="1328" height="966" alt="image" src="https://github.com/user-attachments/assets/d8d23a1c-4aa1-47eb-a0a5-1c15d0310e76" />

Streamlit app for uploading custom images and viewing predictions

Example Predictions:

Bird 🐦 → Predicted as Bird
<img width="2048" height="1280" alt="image" src="https://github.com/user-attachments/assets/740d1434-51f7-47c6-bbd2-216948f78ba7" />


Cat 🐱 → Predicted as Cat
<img width="2048" height="1280" alt="image" src="https://github.com/user-attachments/assets/e68d6615-c2a6-4e3c-a793-48d057958885" />


🚀 Deployment

The trained model (model.h5) is deployed on Streamlit Cloud:
👉 Live App

🧩 Installation
# Clone this repository
git clone https://github.com/Ankitkumarnath/image_classification_using_cifa10-images.git
cd image_classification_using_cifa10-images

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py

📚 Technologies Used

Python

TensorFlow / Keras

NumPy, Matplotlib, scikit-learn

Streamlit (for deployment)

Streamlit app for uploading custom images and viewing predictions

Example Predictions:

Bird 🐦 → Predicted as Bird

Cat 🐱 → Predicted as Cat

🚀 Deployment

The trained model (model.h5) is deployed on Streamlit Cloud:
👉 Live App

🧩 Installation
# Clone this repository
git clone https://github.com/Ankitkumarnath/image_classification_using_cifa10-images.git
cd image_classification_using_cifa10-images

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py

📚 Technologies Used

Python

TensorFlow / Keras

NumPy, Matplotlib, scikit-learn

Streamlit (for deployment)

Streamlit app for uploading custom images and viewing predictions

Example Predictions:

Bird 🐦 → Predicted as Bird

Cat 🐱 → 

🚀 Deployment

The trained model (model.h5) is deployed on Streamlit Cloud:
👉 Live App: https://ankitkumarnath-image-classification-using-cifa10-ima-app-yxdap8.streamlit.app/

🧩 Installation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py

📚 Technologies Used

Python

TensorFlow / Keras

NumPy, Matplotlib, scikit-learn

Streamlit (for deployment)
