import streamlit as st
import base64
import tensorflow as tf
import numpy as np
from groq import Groq
import os
import dotenv

dotenv.load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

# Theme Configuration
st.set_page_config(
    page_title="Plant Sense",
    page_icon='imgs/logo_plant.png',
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for sidebar styling
st.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: #27a440;
        text-align: center;
        margin-bottom: 20px;
    }

    .nav-box {
        background-color: #1e1e1e;
        color: #27a440;
        border: 2px solid #27a440;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        cursor: pointer;
    }

    .nav-box-active {
        background-color: #27a440;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
def sidebar_navigation():
    img_path = 'imgs/logo_plant.png'

    # Encode the image in base64
    img_base64 = base64.b64encode(open(img_path, "rb").read()).decode("utf-8")

    # Display the image with a green glow and smooth edges in the sidebar
    if img_base64:
        st.sidebar.markdown(
            f'''
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_base64}" 
                    style="
                        display: block; 
                        margin: auto; 
                        width: 80%; 
                        max-width: 150px; 
                        border-radius: 15px; 
                        box-shadow: 0 0 20px 5px rgba(39, 164, 64, 0.8);">
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)

    # Initialize session state to track the active page
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Home"

    # Navigation Buttons
    if st.sidebar.button("🏠 Home", key="home_button"):
        st.session_state.active_page = "Home"
    if st.sidebar.button("📖 About", key="about_button"):
        st.session_state.active_page = "About"
    if st.sidebar.button("🩺 Disease Prediction", key="disease_button"):
        st.session_state.active_page = "Disease Prediction"

    return st.session_state.active_page

# Pages
def home_page():
    st.title("🏠 Welcome to the Plant Health Assistant")
    st.write(
        """
        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Disease Prediction** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        """
    )
    st.info("Navigate using the sidebar to explore the features.")


def about_page():
    st.title("📖 About")
    st.write(
        """
        ## About This Application
        The **Plant Sense** is a user-friendly platform for:
        - **Disease Prediction**: Upload the images and know about your plant health by predicting potential diseases.
        - **Health Education**: Learn about symptoms, causes, and treatments.
        - **Easy Navigation**: Intuitive design for users of all levels.

        #### How It Works
        - Navigate to the **Disease Prediction** page.
        - Upload the image
        - And experience the power of our Plant Disease Recognition System!
        """
    )
    st.success("Lets get Staeted!!!")

def disease_prediction_page():
    st.title("🩺 Disease Prediction")
    
    # File uploader for image input
    test_image = st.file_uploader("Choose an Image:")
    
    # Display the uploaded image if the "Show Image" button is pressed
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image first!")

    # Predict button
    if st.button("Predict"):
        if test_image is not None:
            st.write("Our Prediction")
            
            # Predict the result using the model
            result_index = model_prediction(test_image)
            
            # Reading class labels
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            
            # Fetching the predicted class name
            end_result = class_name[result_index]
            st.success(f"Model predicts: {end_result}")

            # Generate explanation and recommendation for the prediction
            suggestion = ""
            
            # Use OpenAI or another API for generating explanation and recommendation
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": '''You are a plant health care assistant. 
                        The user will provide the name of a disease, and you must explain the disease in simple terms 
                        and provide solutions. If the input is "healthy," explain that the plant is healthy 
                        and recommend tips to keep it that way.
                        
                        Format:
                        {
                        Explanation:
                        Recommendation:
                        }'''
                    },
                    {
                        "role": "user",
                        "content": f"This is the user_input: {end_result}"
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )

            for chunk in completion:
                content_piece = chunk.choices[0].delta.content
                if content_piece:
                    suggestion += content_piece
                    print(content_piece, end="")
            
            # Display the generated suggestion
            st.subheader("Our Solution")
            st.write(suggestion)
        else:
            st.warning("Please upload an image first!")


def mock_disease_prediction(symptoms):
    """Mock function to simulate disease prediction."""
    symptom_list = symptoms.split(",")
    if "fever" in symptom_list:
        return "Malaria"
    elif "cough" in symptom_list:
        return "Flu"
    else:
        return "Unknown Condition"

# Main Function
def main():
    # Handle sidebar navigation
    page = sidebar_navigation()

    # Render the selected page
    if page == "Home":
        home_page()

        st.markdown(
        """
        <style>
        .card-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background-color: #1e1e1e;
            color: white;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            width: 300px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 16px rgba(39, 164, 64, 0.5);
        }
        .card-icon {
            font-size: 50px;
            margin-bottom: 15px;
            color: #27a440;
        }
        .card-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .card-desc {
            font-size: 16px;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

        st.markdown(
            """
            <div class="card-container">
                <div class="card">
                    <div class="card-icon">🏡</div>
                    <div class="card-title">Welcome to Plant Sense</div>
                    <div class="card-desc">
                        Learn how to take care of your plants, predict plant diseases, and grow a greener future!
                    </div>
                </div>
                <div class="card">
                    <div class="card-icon">📚</div>
                    <div class="card-title">About</div>
                    <div class="card-desc">
                        Discover how Plant Sense works and explore the features that make it unique.
                    </div>
                </div>
                <div class="card">
                    <div class="card-icon">🔍</div>
                    <div class="card-title">Disease Prediction</div>
                    <div class="card-desc">
                        Enter symptoms and predict potential diseases affecting your plants with AI.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif page == "About":
        about_page()
    elif page == "Disease Prediction":
        disease_prediction_page()

if __name__ == "__main__":
    main()
