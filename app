import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from keras.preprocessing.image import img_to_array
from io import StringIO

class EmotionPredictor:
    def __init__(self, model, df):
        self.model = model
        self.df = df

    def predict_emotion(self, image):
        processed_image = self.prepare_image(image)
        prediction = self.model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)
        emotion = self.index_to_emotion(predicted_class[0], "Unknown Emotion")

        if emotion == "Disgusted":
            results = self.df[self.df['Emotion'].isin(['Energetic', 'Happy', 'Happily In Love'])].sample(5, replace=False)
        elif emotion == "Angry":
            results = self.df[self.df['Emotion'].isin(['Romantic'])].sample(5, replace=False)
        elif emotion == "Fearful":
            results = self.df[self.df['Emotion'].isin(['Energetic', 'Intense'])].sample(5, replace=False)
        elif emotion == "Happy":
            results = self.df[self.df['Emotion'].isin(['Sad', 'Intense'])].sample(5, replace=False)
        elif emotion == "Sad":
            results = self.df[self.df['Emotion'].isin(['Happy', 'Happily In Love', 'Energetic'])].sample(5, replace=False)
        elif emotion == "Surprised":
            results = self.df[self.df['Emotion'].isin(['Energetic', 'Happy', 'Sad'])].sample(5, replace=False)
        elif emotion == "Neutral":
            results = self.df[self.df['Emotion'].isin(['Energetic', 'Happy', 'Sad', 'Happily In Love'])].sample(5, replace=False)

        formatted_result_str = f"Predicted Emotion: {emotion}\n\n"
        formatted_result_str += "".join(
          f"**Title:** {row['Song Title']}\n"
          f"**Artist:** {row['Artist']}\n"
          f"**Album:** {row['Album']}\n"
          f"**Release Date:** {row['Release Date']}\n\n"
          for _, row in results.iterrows()
        )

        return formatted_result_str

    def prepare_image(self, img_pil):
        img = img_pil.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def index_to_emotion(self, index, default_emotion):
        emotion_dict = {
            3: 'Happy',
            5: 'Sad',
            0: 'Angry',
            6: 'Surprised',
            4: 'Neutral',
            1: 'Disgusted',
            2: 'Fearful'
        }
        return emotion_dict.get(index, default_emotion)

# Load your model and data
df = pd.read_csv('songs_new.csv')  # Update the path as necessary
model = ...  # Load your trained model here

predictor = EmotionPredictor(model, df)

# Streamlit app
st.title("Moodify")
st.write("Upload an image to predict the emotion and get song recommendations based on the emotion detected.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    result = predictor.predict_emotion(image)
    st.write(result)
