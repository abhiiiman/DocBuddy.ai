# importing the required libraries here.
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
import requests
import json
import ast
from dotenv import load_dotenv
import os
import warnings
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from typing import Generator
from groq import Groq
from langdetect import detect
from translate import Translator

warnings.filterwarnings("ignore")

load_dotenv()

# Load Firebase credentials from Streamlit secrets
firebase_creds = {
    "type": st.secrets["firebase"]["type"],
    "project_id": st.secrets["firebase"]["project_id"],
    "private_key_id": st.secrets["firebase"]["private_key_id"],
    "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
    "client_email": st.secrets["firebase"]["client_email"],
    "client_id": st.secrets["firebase"]["client_id"],
    "auth_uri": st.secrets["firebase"]["auth_uri"],
    "token_uri": st.secrets["firebase"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
    "universe_domain": st.secrets["firebase"]["universe_domain"]
}

# init firebase app here.
cred = credentials.Certificate(firebase_creds)
try:
    firebase_admin.get_app()
except ValueError as e:
    firebase_admin.initialize_app(cred)

# setting up the page header here.
hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

# setting up the page config here.
st.set_page_config(
    page_title="DocBuddy.ai",
    page_icon=r"favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/abhiiiman',
        'Report a bug': "https://www.github.com/abhiiiman",
        'About': "## Your Personalized AI enabled Doctor 👨🏻‍⚕️"
    }
)

# removing all the default streamlit configs here
st.markdown(hide_st_style, unsafe_allow_html=True)

# loading the dataset here
symptom_data = pd.read_csv("symptoms_df.csv")
precautions_data = pd.read_csv("precautions_df.csv")
workout_data = pd.read_csv("workout_df.csv")
desc_data = pd.read_csv("description.csv")
diets_data = pd.read_csv("diets.csv")
medication_data = pd.read_csv("medications.csv")

# Replace 'nan' string and np.nan with None for consistency
precautions_data.replace('nan', None, inplace=True)
precautions_data = precautions_data.where(pd.notnull(precautions_data), None)

symptoms_dict = {
    'itching': 0,
    'skin_rash': 1,
    'nodal_skin_eruptions': 2,
    'continuous_sneezing': 3,
    'shivering': 4,
    'chills': 5,
    'joint_pain': 6,
    'stomach_pain': 7,
    'acidity': 8,
    'ulcers_on_tongue': 9,
    'muscle_wasting': 10,
    'vomiting': 11,
    'burning_micturition': 12,
    'spotting_urination': 13,
    'fatigue': 14,
    'weight_gain': 15,
    'anxiety': 16,
    'cold_hands_and_feets': 17,
    'mood_swings': 18,
    'weight_loss': 19,
    'restlessness': 20,
    'lethargy': 21,
    'patches_in_throat': 22,
    'irregular_sugar_level': 23,
    'cough': 24,
    'high_fever': 25,
    'sunken_eyes': 26,
    'breathlessness': 27,
    'sweating': 28,
    'dehydration': 29,
    'indigestion': 30,
    'headache': 31,
    'yellowish_skin': 32,
    'dark_urine': 33,
    'nausea': 34,
    'loss_of_appetite': 35,
    'pain_behind_the_eyes': 36,
    'back_pain': 37,
    'constipation': 38,
    'abdominal_pain': 39,
    'diarrhoea': 40,
    'mild_fever': 41,
    'yellow_urine': 42,
    'yellowing_of_eyes': 43,
    'acute_liver_failure': 44,
    'fluid_overload': 45,
    'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47,
    'malaise': 48,
    'blurred_and_distorted_vision': 49,
    'phlegm': 50,
    'throat_irritation': 51,
    'redness_of_eyes': 52,
    'sinus_pressure': 53,
    'runny_nose': 54,
    'congestion': 55,
    'chest_pain': 56,
    'weakness_in_limbs': 57,
    'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60,
    'bloody_stool': 61,
    'irritation_in_anus': 62,
    'neck_pain': 63,
    'dizziness': 64,
    'cramps': 65,
    'bruising': 66,
    'obesity': 67,
    'swollen_legs': 68,
    'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70,
    'enlarged_thyroid': 71,
    'brittle_nails': 72,
    'swollen_extremeties': 73,
    'excessive_hunger': 74,
    'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76,
    'slurred_speech': 77,
    'knee_pain': 78,
    'hip_joint_pain': 79,
    'muscle_weakness': 80,
    'stiff_neck': 81,
    'swelling_joints': 82,
    'movement_stiffness': 83,
    'spinning_movements': 84,
    'loss_of_balance': 85,
    'unsteadiness': 86,
    'weakness_of_one_body_side': 87,
    'loss_of_smell': 88,
    'bladder_discomfort': 89,
    'foul_smell_of_urine': 90,
    'continuous_feel_of_urine': 91,
    'passage_of_gases': 92,
    'internal_itching': 93,
    'toxic_look_(typhos)': 94,
    'depression': 95,
    'irritability': 96,
    'muscle_pain': 97,
    'altered_sensorium': 98,
    'red_spots_over_body': 99,
    'belly_pain': 100,
    'abnormal_menstruation': 101,
    'dischromic_patches': 102,
    'watering_from_eyes': 103,
    'increased_appetite': 104,
    'polyuria': 105,
    'family_history': 106,
    'mucoid_sputum': 107,
    'rusty_sputum': 108,
    'lack_of_concentration': 109,
    'visual_disturbances': 110,
    'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112,
    'coma': 113,
    'stomach_bleeding': 114,
    'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117,
    'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119,
    'palpitations': 120,
    'painful_walking': 121,
    'pus_filled_pimples': 122,
    'blackheads': 123,
    'scurring': 124,
    'skin_peeling': 125,
    'silver_like_dusting': 126,
    'small_dents_in_nails': 127,
    'inflammatory_nails': 128,
    'blister': 129,
    'red_sore_around_nose': 130,
    'yellow_crust_ooze': 131
}
diseases_list = {
    15: 'Fungal infection',
    4: 'Allergy',
    16: 'GERD',
    9: 'Chronic cholesterol',
    14: 'Drug Reaction',
    33: 'Peptic ulcer disease',
    1: 'AIDS',
    12: 'Diabetes',
    17: 'Gastroenteritis',
    6: 'Bronchial Asthma',
    23: 'Hypertension',
    30: 'Migraine',
    7: 'Cervical spondylosis',
    32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice',
    29: 'Malaria',
    8: 'Chicken pox',
    11: 'Dengue',
    37: 'Typhoid',
    40: 'Hepatitis A',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    3: 'Alcoholic hepatitis',
    36: 'Tuberculosis',
    10: 'Common Cold',
    34: 'Pneumonia',
    13: 'Dimorphic hemorrhoids (piles)',
    18: 'Heart attack',
    39: 'Varicose veins',
    26: 'Hypothyroidism',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    31: 'Osteoarthritis',
    5: 'Arthritis',
    0: '(vertigo) Paroxysmal Positional Vertigo',
    2: 'Acne',
    38: 'Urinary tract infection',
    35: 'Psoriasis',
    27: 'Impetigo'
}

# Initialize session state for the data to generate the report
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if 'disease' not in st.session_state:
    st.session_state.disease = None
if 'description' not in st.session_state:
    st.session_state.description = None
if 'precautions' not in st.session_state:
    st.session_state.precautions = None
if 'workout' not in st.session_state:
    st.session_state.workout = None
if 'diets' not in st.session_state:
    st.session_state.diets = None
if 'medications' not in st.session_state:
    st.session_state.medications = None


def generate_report(name, age, disease, description, precautions, workouts, diets, medications, file_path):
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    styleN = styles['BodyText']
    styleH = styles['Heading1']

    # getting the current date and time here
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d")

    # Title
    title = Paragraph("DocBuddy Health Report", styleH)
    story = [title, Spacer(1, 12)]

    # Personal details
    details = Paragraph("Patient Details ", styleH)
    story = [details, Spacer(1, 10)]
    name = Paragraph(f"Patient Name : <b>{name.title()}</b>", styleN)
    story.append(name)
    story.append(Spacer(1, 10))
    age = Paragraph(f"Patient Age : <b>{age} Years</b>", styleN)
    story.append(age)
    story.append(Spacer(1, 10))
    date = Paragraph(f"Report Generated On : <b>{current_time}</b>", styleN)
    story.append(date)
    story.append(Spacer(1, 12))

    # Predicted Disease
    disease_paragraph = Paragraph(f"Predicted Disease : <b>{disease.title()}</b>", styleN)
    story.append(disease_paragraph)
    story.append(Spacer(1, 12))

    # Description
    description_paragraph = Paragraph(f"Description : <b>{description}</b>", styleN)
    story.append(description_paragraph)
    story.append(Spacer(1, 12))

    # Precautions
    precautions_paragraph = Paragraph("Precautions : ", styleH)
    story.append(precautions_paragraph)
    story.append(Spacer(1, 12))
    precautions_list = ListFlowable([ListItem(Paragraph(p, styleN)) for p in precautions if p is not None],
                                    bulletType='bullet')
    story.append(precautions_list)
    story.append(Spacer(1, 12))

    # Workouts
    workouts_paragraph = Paragraph("Recommendations : ", styleH)
    story.append(workouts_paragraph)
    story.append(Spacer(1, 12))
    workouts_list = ListFlowable([ListItem(Paragraph(w, styleN)) for w in workouts], bulletType='bullet')
    story.append(workouts_list)
    story.append(Spacer(1, 12))

    # Diets
    diets_paragraph = Paragraph("Diets :", styleH)
    story.append(diets_paragraph)
    story.append(Spacer(1, 12))
    diets_list = ListFlowable([ListItem(Paragraph(d, styleN)) for d in diets], bulletType='bullet')
    story.append(diets_list)
    story.append(Spacer(1, 12))

    # Medications
    medications_paragraph = Paragraph("Medications :", styleH)
    story.append(medications_paragraph)
    story.append(Spacer(1, 12))
    medications_list = ListFlowable([ListItem(Paragraph(m, styleN)) for m in medications], bulletType='bullet')
    story.append(medications_list)
    story.append(Spacer(1, 12))

    # Build the PDF
    doc.build(story)
    print(f"PDF report generated successfully: {file_path}")


# Function to predict the disease
def get_predicted_values(patient_symptoms):
    st.session_state.predicted = True
    model = pickle.load(open('model.pkl', 'rb'))
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        # making the index value 1 for that respective disease.
        input_vector[symptoms_dict[symptom]] = 1
    return diseases_list[model.predict([input_vector])[0]]


def get_desc(predicted_value):
    predicted_description = desc_data[desc_data["Disease"] == predicted_value]["Description"].values[0]
    return predicted_description


def get_precautions(predicted_value):
    predicted_precaution = precautions_data[precautions_data['Disease'] == predicted_value].values[0][2:]
    return predicted_precaution


def print_precautions(p):
    c = 1
    for j in range(len(p)):
        if p[j] is not None:
            st.write(f"Precaution {c}. -> {p[j].title()}.")
            c += 1


def print_workout(w):
    c = 1
    for j in range(len(w)):
        st.write(f"Workout {c}. -> {w[j].title()}.")
        c += 1


def get_medication(predicted_value):
    med = medication_data[medication_data['Disease'] == predicted_value]['Medication'].values[0]
    return ast.literal_eval(med)


def get_workout(predicted_value):
    work = workout_data[workout_data['disease'] == predicted_value]["workout"].values
    return work


def get_diet(predicted_value):
    diet = diets_data[diets_data['Disease'] == predicted_value]['Diet'].values[0]
    return ast.literal_eval(diet)


def account():
    st.image(r"Login-DocBuddy.png")
    st.title("Welcome to DocBuddy 🩺")

    # Create session state variables
    if "user_mail" not in st.session_state:
        st.session_state.user_mail = ""
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "signedOut" not in st.session_state:
        st.session_state.signedOut = False
    if "signOut" not in st.session_state:
        st.session_state.signOut = False
    if "userName" not in st.session_state:
        st.session_state.userName = ""

    def logout():
        st.session_state.predicted = False
        st.session_state.signOut = False
        st.session_state.signedOut = False
        st.session_state.user_name = ""
        st.session_state.user_mail = ""

    if not st.session_state['signedOut']:
        choice = st.selectbox("Login / Sign Up ➕", ["Login", "Sign Up"])
        if choice == "Login":
            email = st.text_input("Email 📧").strip()
            password = st.text_input("Password 🔑", type="password")
            login_submit = st.button("Login")

            # Authenticate the user
            if login_submit:
                print("User Login")
                print(f"Email : {email}, Password : {password}")

                def auth_user():
                    try:
                        # Firebase Auth REST API endpoint for sign-in with email and password
                        url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
                        api_key = st.secrets["FIREBASE_API_KEY"]  # Replace with your Firebase Web API Key

                        payload = json.dumps({
                            "email": email,
                            "password": password,
                            "returnSecureToken": True
                        })

                        headers = {
                            'Content-Type': 'application/json'
                        }

                        response = requests.post(f"{url}?key={api_key}", headers=headers, data=payload)

                        if response.status_code == 200:
                            user_data = response.json()
                            user_name = user_data["localId"]
                            st.success(f"Welcome {user_name}! Login Successful 🥳")
                            st.balloons()

                            # Update session state variables
                            st.session_state.user_mail = email
                            st.session_state.user_name = user_name
                            st.session_state.signOut = True
                            st.session_state.signedOut = True
                        else:
                            st.warning("Invalid Username/Password, Try Again! ⚠️")

                    except Exception as e:
                        st.warning("Invalid Username/Password, Try Again! ⚠️")

                auth_user()
        else:
            email = st.text_input("Email 📧").strip()
            password = st.text_input("Password 🔑", type="password")
            userName = st.text_input("Create your Unique Username 👤")
            signup_submit = st.button("Create My Account")

            if signup_submit:
                print("User Sign Up")
                print(f"Email : {email}, Password : {password}, UserName : {userName}")

                def create_account():
                    if not userName:
                        st.warning("Username cannot be empty! ⚠️")
                        return

                    try:
                        user = auth.get_user_by_email(email)
                        st.warning(f"{user.uid} already exists! Try Login instead.")
                    except firebase_admin.auth.UserNotFoundError:
                        user = auth.create_user(email=email, password=password, uid=userName)
                        st.success("Account Created Successfully 🥳")
                        st.balloons()
                        st.write("Please Login Now!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

                create_account()

    if st.session_state.signOut:
        st.text(f"UserName: {st.session_state.user_name}")
        st.text(f"UserMail: {st.session_state.user_mail}")
        st.button("Log Out", on_click=logout)


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 0">{emoji}</span>',
        unsafe_allow_html=True,
    )


def detect_lang(text: str) -> str:
    detected_lang = detect(text)
    return detected_lang


def get_translation(src, target_lang):
    translator = Translator(to_lang=target_lang)
    translation = translator.translate(src)
    return translation


def medical_chatbot():
    # connecting to groq cloud here.
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    # Initialize chat history and selected model
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Define model details
    models = {
        "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
        "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
        "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    }

    # Layout for model selection and max_tokens slider
    col_1, col_2 = st.columns(2)

    with col_1:
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=2  # Default to llama3
        )

    # Detect model change and clear chat history if model has changed
    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    max_tokens_range = models[model_option]["tokens"]

    with col_2:
        # Adjust max_tokens slider dynamically based on the selected model
        max_tokens = st.slider(
            "Max Tokens:",
            min_value=512,  # Minimum value to allow some flexibility
            max_value=max_tokens_range,
            # Default value or max allowed if less
            value=min(32768, max_tokens_range),
            step=512,
            help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
        )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    if prompt := st.chat_input("Enter your texts here and chat..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)

        # Fetch response from Groq API
        full_response = ""
        try:
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages
                ],
                max_tokens=max_tokens,
                stream=True
            )

            # Use the generator function with st.write_stream
            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages
        if isinstance(full_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response})


with st.sidebar:
    selected = option_menu(
        menu_title="DocBuddy.ai",
        options=["Home", "Recommendations", "Generate Report", "Chat With Me", "WorkFlow", "Account"],
        icons=["house", "magic", "book", "chat", "activity", "gear"],
        menu_icon="app-indicator",
        default_index=0,
    )

# ========= HOME TAB =========
if selected == "Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title('DocBuddy.ai 🩺')
        st.header("Your Personalized 🪄 Doctor Buddy 👨🏻‍⚕️")
        st.divider()
        st.header("About 👨🏻‍⚕️🩺")
        st.markdown('''
        DocBuddy.ai is an innovative application designed to revolutionize the way you manage your health.
        Our intelligent machine learning model accurately predicts potential diseases based on your symptoms, 
        providing you with timely insights and empowering you to take control of your well-being. 🏥✨
        ''')
        st.markdown('''
        ### Join the `DocBuddy.ai` Community 😃
        Take charge of your health with DocBuddy.ai! 🌟 Download the app today and experience the future of health management.
        Connect with us on social media and be part of a community that values well-being and proactive health management.
        ''')
        st.markdown("_Stay healthy, stay informed, and let DocBuddy.ai be your trusted health companion_ 💪❤️")

        st.markdown("#### Get Started Now!")
        st.markdown("DocBuddy is here to help you live your healthiest life!")

    with col2:
        st.image(r"DocBuddy-Home.png")

# ========= WORKFLOW TAB =========
elif selected == "WorkFlow":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title('DocBuddy.ai WorkFlow ⛑️')
        st.header("How Does It Work? 🤔")
        st.divider()
        st.subheader("1️⃣ Symptom Input 🤒")
        st.markdown('''
            * Simply enter the symptoms you're experiencing into our user-friendly interface. Whether it's a 
            headache, fever, or any other discomfort, DocBuddy.ai is here to help.
        ''')
        st.subheader("2️⃣ Disease Prediction 🔍")
        st.markdown('''
            * Our sophisticated machine learning model analyzes the symptoms and predicts the most likely diseases. This fast and efficient process ensures you get accurate information without the wait.
        ''')
        st.subheader("3️⃣ Detailed Descriptions 📖")
        st.markdown('''
            * Once a disease is predicted, DocBuddy.ai provides a comprehensive description of the condition. You'll learn about its causes, symptoms, and potential treatments, helping you understand your health better.
        ''')
        st.subheader("4️⃣ Personalized Recommendations 🌿💊")
        st.markdown('''
            DocBuddy.ai goes beyond mere diagnosis. It offers personalized recommendations for:
            * `Medicines` : Find out which over-the-counter or prescription medicines can help alleviate your 
            symptoms.
            * `Workout Plans` : Get tailored exercise routines to boost your overall health and manage your condition.
            * `Diets` : Discover dietary suggestions to support your recovery and maintain a balanced lifestyle.
            * `Precautions` : Learn about preventive measures to avoid aggravating your condition and protect your 
            health.
        ''')

    with col2:
        st.image(r"DocBuddy-WorkFlow-Tab.png")

# ========= Accounts TAB =========
elif selected == "Account":
    account()

# ========= Recommendations TAB =========
elif selected == "Recommendations":
    col1, col2 = st.columns([3, 1])
    with col1:
        # Check if the user is logged in using session state variables
        if st.session_state.get("signedOut", False):
            st.title(f"Welcome {st.session_state.user_name} 🎉")
            st.header("DocBuddy Recommendation Center 🔮")
            st.divider()
            symptoms_list = [
                'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                'joint_pain',
                'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
                'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
                'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
                'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
                'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
                'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
                'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
                'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
                'phlegm', 'throat_irritation',
                'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
                'fast_heart_rate',
                'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
                'dizziness', 'cramps',
                'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
                'enlarged_thyroid',
                'brittle_nails',
                'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
                'slurred_speech', 'knee_pain',
                'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
                'spinning_movements', 'loss_of_balance',
                'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
                'foul_smell_of_urine',
                'continuous_feel_of_urine',
                'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
                'muscle_pain',
                'altered_sensorium',
                'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches',
                'watering_from_eyes',
                'increased_appetite',
                'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
                'visual_disturbances', 'receiving_blood_transfusion',
                'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
                'history_of_alcohol_consumption',
                'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking',
                'pus_filled_pimples',
                'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
                'inflammatory_nails', 'blister',
                'red_sore_around_nose', 'yellow_crust_ooze'
            ]
            symptoms = st.multiselect("#### Select Patient's Symptoms below 👇🏻", symptoms_list)
            predict_button = st.button("Predict Disease 🔮")
            if predict_button:
                if len(symptoms) == 0:
                    st.warning("Please Select Symptoms first from the Dropdown List then Predict Disease ⚠️")
                else:
                    disease = get_predicted_values(symptoms)
                    print(disease)
                    st.session_state.disease = disease
                    st.session_state.description = get_desc(disease)
                    st.session_state.precautions = get_precautions(disease)
                    st.session_state.workout = get_workout(disease)
                    st.session_state.diets = get_diet(disease)
                    st.session_state.medications = get_medication(disease)
                    st.session_state.predicted = True

                    st.subheader(f"You have `{st.session_state.disease}` 🤒")
                    st.write(f"👉🏻 {st.session_state.description}")

                    st.divider()

                    col3, col4, col5 = st.columns([3, 4, 3])

                    with col3:
                        st.subheader("⚠️ Precautions")
                        for precaution in st.session_state.precautions:
                            if precaution is not None:
                                st.write(f"🫵🏻 {precaution.title()}")

                    with col4:
                        st.subheader("✨ Recommendations")
                        for workout in st.session_state.workout:
                            st.write(f"👉🏻 {workout.title()}")

                    with col3:
                        st.subheader("🍚 Diets")
                        for diet in st.session_state.diets:
                            st.write(f"➕ {diet.title()}")

                    with col5:
                        st.subheader("💊 Medications")
                        for med in st.session_state.medications:
                            st.write(f"✔️ {med.title()}")
        else:
            st.title("Please Login First ⚠️")
            st.subheader("You are not logged in!")
            st.markdown("* Please go back to the Account section.")
            st.markdown("* Then go to the Login Page and Login Yourself.")
    with col2:
        st.image(r"DocBuddy-Recommendations.png")

# ========= Report Generation TAB =========
elif selected == "Generate Report":
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.session_state.get("signedOut", False):
            st.title(f"Welcome {st.session_state.user_name} 🎉")
            st.header("DocBuddy Medical Report Generation 📃")
            st.divider()
            col3, col4 = st.columns([2, 2])
            with col3:
                name = st.text_input("Enter the patient Name below", placeholder="Name")
            with col4:
                age = st.number_input("Enter the patient Age below", placeholder="Age", value=None, min_value=1,
                                      max_value=100)
            generate = st.button("Generate DocBuddy Report ✨")
            st.warning("⚠️ This is an automated AI generated report prepared by DocBuddy.ai")
            st.write("It's always better to see a Doctor and consult them before taking any step further!")
            st.divider()
            if generate:
                if st.session_state.predicted:
                    if name and age:
                        generate_report(
                            name,
                            age,
                            disease=st.session_state.disease,
                            description=st.session_state.description,
                            precautions=st.session_state.precautions,
                            workouts=st.session_state.workout,
                            diets=st.session_state.diets,
                            medications=st.session_state.medications,
                            file_path=f"DocBuddy_{name.title()}_Report.pdf"
                        )
                        with open(f"DocBuddy_{name.title()}_Report.pdf", "rb") as file:
                            st.download_button(
                                label="Download Generated Report ✅",
                                data=file,
                                file_name=f"DocBuddy_{name.title()}_Report.pdf",
                                mime="pdf",
                            )
                    else:
                        st.warning("⚠️ Please enter correct Name/Age to proceed")
                else:
                    st.warning("⚠️ It seems like you haven't got your DocBuddy Recommendations!")
                    st.markdown("* Go to `Recommendations` tab first on the top left sidebar.")
                    st.markdown("* Get your `Recommendations` there first.")
                    st.markdown("* Then comeback here and apply for `Report Generation`.")
        else:
            st.title("Please Login First ⚠️")
            st.subheader("Log in first, to Generate Report")
            st.markdown("* Please go back to the Account section.")
            st.markdown("* Then go to the Login Page and Login Yourself.")
    with col2:
        st.image(r"DocBuddy-Generate-Report.png")

# ========= Chat with me TAB =========
elif selected == "Chat With Me":
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.session_state.get("signedOut", False):
            st.markdown(f"#### Welcome, {st.session_state.user_name} 🎉")
            st.markdown("""
                # :rainbow[Chat With DocBuddy.ai 🗨️]
            """)
            # icon("🧑🏻‍⚕️")
            st.subheader("Medical HealthCare ChatBot `Premium`", divider="rainbow", anchor=False)
            medical_chatbot()
        else:
            st.title("Please Login First ⚠️")
            st.subheader("Start Chatting with DocBuddy.ai 🗨️")
            st.markdown("* Please go back to the Account section.")
            st.markdown("* Then go to the Login Page and Login Yourself.")
    with col2:
        st.image(r"DOcBuddy-Chat-With-Me.png")
