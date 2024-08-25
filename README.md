# Moodify

**Moodify** is a machine learning-based song recommendation system that predicts a user's emotion using facial expression recognition and recommends songs accordingly. The system uses deep learning models such as VGG16 and ResNet50 for emotion detection, which is then used to suggest songs from a curated dataset. The project is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/ankush76/Moodify_Beta1/tree/main) with an interactive web interface built using Gradio.

## Features
- **Emotion Detection**: Utilizes Convolutional Neural Networks (CNN) models like VGG16 and ResNet50 to classify facial emotions.
- **Song Recommendation**: Recommends songs based on the predicted emotion, using a custom-trained model with a songs dataset.
- **Web Interface**: Deployed on Hugging Face Spaces with a user-friendly web interface built using Gradio.
- **Accurate Performance**: Achieves an accuracy of 66% on emotion detection using the FER-2013 dataset, optimized through data augmentation and model fine-tuning.

## Demo
Try the live demo on Hugging Face Spaces: [Moodify Demo](https://huggingface.co/spaces/ankush76/Moodify_Beta1/tree/main)

## Tech Stack
- **TensorFlow**: Used for building and training the CNN models for emotion detection.
- **Gradio**: Created an interactive web interface for easy use and deployment of the model.
- **VGG16 and ResNet50**: Pre-trained models fine-tuned for the task of emotion recognition.
- **Hugging Face Spaces**: Used to deploy the web application for public access.

## Project Structure
- `notebooks/`: Contains Jupyter notebooks used for model development, training, and testing.
- `models/`: Pre-trained models used for emotion detection.
- `app.py`: The Gradio-powered web application script for deploying the model.
- `data/`: Includes the songs dataset and necessary resources for the recommendation system.

## How It Works
1. **Emotion Detection**: A user uploads an image, and the model classifies the facial expression into one of the predefined emotion categories (e.g., happy, sad, angry).
2. **Song Recommendation**: Based on the detected emotion, the system recommends a song that aligns with the user's mood.
3. **Interactive UI**: Users interact with the model through a web interface that allows them to upload images and receive song recommendations.
