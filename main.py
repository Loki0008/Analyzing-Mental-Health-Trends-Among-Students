

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Student Depression Prediction API",
    description="API for predicting depression status among students",
    version="1.0.0"
)


# ============================================================
# 2. LOAD SAVED ML FILES
# ============================================================

model = joblib.load("random_forest_model.pkl")
selector = joblib.load("selector.pkl")
scale = joblib.load("scale.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ============================================================
# 3. CITY VALUES FROM YOUR ACTUAL DATASET
# ============================================================

cities = [
    "Agra",
    "Ahmedabad",
    "Bangalore",
    "Bhopal",
    "Chennai",
    "Delhi",
    "Faridabad",
    "Ghaziabad",
    "Hyderabad",
    "Indore",
    "Jaipur",
    "Kalyan",
    "Kanpur",
    "Kolkata",
    "Lucknow",
    "Ludhiana",
    "Meerut",
    "Mumbai",
    "Nagpur",
    "Nashik",
    "Patna",
    "Pune",
    "Rajkot",
    "Srinagar",
    "Surat",
    "Thane",
    "Vadodara",
    "Varanasi",
    "Vasai-Virar",
    "Visakhapatnam"
]


# ============================================================
# 4. INPUT DATA MODEL
# ============================================================

class StudentData(BaseModel):

    Gender: str
    Age: float
    City: str

    Academic_Pressure: float
    CGPA: float
    Study_Satisfaction: float

    Sleep_Duration: str
    Dietary_Habits: str
    Degree: str

    suicidal_thoughts: str

    Work_Study_Hours: float
    Financial_Stress: float

    Family_History_of_Mental_Illness: str


# ============================================================
# 5. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Student Depression Prediction API is running",
        "status": "success"
    }


# ============================================================
# 6. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: StudentData):

    # --------------------------------------------------------
    # STEP 1: Convert API input into a dictionary
    # --------------------------------------------------------

    input_data = data.model_dump()


    # --------------------------------------------------------
    # STEP 2: VALIDATE INPUT VALUES
    # --------------------------------------------------------

    if input_data["Gender"] not in ["Male", "Female"]:
        return {
            "error": "Gender must be 'Male' or 'Female'"
        }

    if input_data["City"] not in cities:
        return {
            "error": f"Invalid city. Please use one of the cities from the dataset."
        }

    if input_data["suicidal_thoughts"] not in ["Yes", "No"]:
        return {
            "error": "suicidal_thoughts must be 'Yes' or 'No'"
        }

    if input_data["Family_History_of_Mental_Illness"] not in ["Yes", "No"]:
        return {
            "error": "Family_History_of_Mental_Illness must be 'Yes' or 'No'"
        }

    if input_data["Dietary_Habits"] not in [
        "Healthy",
        "Moderate",
        "Unhealthy"
    ]:
        return {
            "error": "Dietary_Habits must be 'Healthy', 'Moderate', or 'Unhealthy'"
        }

    if input_data["Sleep_Duration"] not in [
        "'Less than 5 hours'",
        "'5-6 hours'",
        "'7-8 hours'",
        "'More than 8 hours'",
        "Others"
    ]:
        return {
            "error": "Invalid Sleep_Duration value"
        }


    # --------------------------------------------------------
    # STEP 3: GENDER LABEL ENCODING
    #
    # Your notebook used LabelEncoder.
    # Actual dataset values:
    # Female, Male
    #
    # LabelEncoder gives:
    # Female = 0
    # Male   = 1
    # --------------------------------------------------------

    gender_mapping = {
        "Female": 0,
        "Male": 1
    }

    gender = gender_mapping[input_data["Gender"]]


    # --------------------------------------------------------
    # STEP 4: SUICIDAL THOUGHTS LABEL ENCODING
    #
    # No  = 0
    # Yes = 1
    # --------------------------------------------------------

    suicidal_mapping = {
        "No": 0,
        "Yes": 1
    }

    suicidal_thoughts = suicidal_mapping[
        input_data["suicidal_thoughts"]
    ]


    # --------------------------------------------------------
    # STEP 5: FAMILY HISTORY LABEL ENCODING
    #
    # No  = 0
    # Yes = 1
    # --------------------------------------------------------

    family_history_mapping = {
        "No": 0,
        "Yes": 1
    }

    family_history = family_history_mapping[
        input_data["Family_History_of_Mental_Illness"]
    ]


    # --------------------------------------------------------
    # STEP 6: SLEEP DURATION MAPPING
    #
    # Same mapping used in your notebook
    # --------------------------------------------------------

    sleep_mapping = {

        "'Less than 5 hours'": 0,

        "'5-6 hours'": 1,

        "'7-8 hours'": 2,

        "'More than 8 hours'": 3,

        "Others": 4
    }

    sleep_duration = sleep_mapping[
        input_data["Sleep_Duration"]
    ]


    # --------------------------------------------------------
    # STEP 7: DIETARY HABITS MAPPING
    #
    # Same mapping used in your notebook
    # --------------------------------------------------------

    dietary_mapping = {

        "Unhealthy": 0,

        "Moderate": 1,

        "Healthy": 2
    }

    dietary_habits = dietary_mapping[
        input_data["Dietary_Habits"]
    ]


    # --------------------------------------------------------
    # STEP 8: DEGREE GROUPING
    #
    # Same grouping used in your notebook
    # --------------------------------------------------------

    bachelor = [
        "B.Pharm",
        "BSc",
        "BA",
        "BHM",
        "BCA",
        "B.Ed",
        "LLB",
        "BE",
        "M.Ed",
        "B.Com",
        "B.Arch",
        "B.Tech",
        "BBA"
    ]

    masters = [
        "M.Tech",
        "M.Ed",
        "MSc",
        "M.Pharm",
        "MCA",
        "MA",
        "MD",
        "MBA",
        "MBBS",
        "M.Com",
        "LLM",
        "ME",
        "MHM",
        "Others"
    ]

    doctorate = [
        "PhD"
    ]

    school = [
        "'Class 12'"
    ]


    if input_data["Degree"] in bachelor:

        degree_group = "Under-Graduate"

    elif input_data["Degree"] in masters:

        degree_group = "Post-Graduate"

    elif input_data["Degree"] in doctorate:

        degree_group = "Doctorate"

    elif input_data["Degree"] in school:

        degree_group = "School"

    else:

        return {
            "error": "Invalid Degree value"
        }


    # --------------------------------------------------------
    # STEP 9: DEGREE NUMERICAL MAPPING
    #
    # Same mapping used in your notebook
    # --------------------------------------------------------

    degree_mapping = {

        "School": 0,

        "Under-Graduate": 1,

        "Post-Graduate": 2,

        "Doctorate": 3
    }

    degree = degree_mapping[degree_group]


    # --------------------------------------------------------
    # STEP 10: CREATE BASIC FEATURE DATAFRAME
    # --------------------------------------------------------

    processed_data = {

        "Gender": gender,

        "Age": input_data["Age"],

        "Academic Pressure": input_data["Academic_Pressure"],

        "CGPA": input_data["CGPA"],

        "Study Satisfaction": input_data["Study_Satisfaction"],

        "Sleep Duration": sleep_duration,

        "Dietary Habits": dietary_habits,

        "Degree": degree,

        "suicidal thoughts ?": suicidal_thoughts,

        "Work/Study Hours": input_data["Work_Study_Hours"],

        "Financial Stress": input_data["Financial_Stress"],

        "Family History of Mental Illness": family_history
    }


    # --------------------------------------------------------
    # STEP 11: CREATE CITY ONE-HOT ENCODING
    #
    # Your notebook used:
    #
    # pd.get_dummies(df['City'], dtype=int, prefix='City')
    #
    # Therefore:
    #
    # City_Mumbai = 1
    # City_Chennai = 0
    # etc.
    # --------------------------------------------------------

    for city in cities:

        city_column = f"City_{city}"

        if city_column in feature_columns:

            if input_data["City"] == city:

                processed_data[city_column] = 1

            else:

                processed_data[city_column] = 0


    # --------------------------------------------------------
    # STEP 12: CREATE DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame([processed_data])


    # --------------------------------------------------------
    # STEP 13: MAKE SURE ALL TRAINING FEATURES EXIST
    #
    # feature_columns.pkl contains the exact X.columns
    # from your notebook.
    # --------------------------------------------------------

    for column in feature_columns:

        if column not in input_df.columns:

            input_df[column] = 0


    # --------------------------------------------------------
    # STEP 14: ARRANGE FEATURES IN EXACT TRAINING ORDER
    # --------------------------------------------------------

    input_df = input_df[feature_columns]


    # --------------------------------------------------------
    # STEP 15: SELECT THE SAME 20 FEATURES
    #
    # selector.pkl was fitted using:
    #
    # SelectKBest(score_func=chi2, k=20)
    # --------------------------------------------------------

    selected_data = selector.transform(input_df)


    # --------------------------------------------------------
    # STEP 16: SCALE THE SELECTED FEATURES
    #
    # Same StandardScaler from your notebook
    # --------------------------------------------------------

    scaled_data = scale.transform(selected_data)


    # --------------------------------------------------------
    # STEP 17: MAKE PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(scaled_data)[0]


    # --------------------------------------------------------
    # STEP 18: GET PREDICTION PROBABILITY
    # --------------------------------------------------------

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(scaled_data)[0]

        probability = float(np.max(probabilities))


    # --------------------------------------------------------
    # STEP 19: RETURN RESULT
    # --------------------------------------------------------

    return {

        "prediction": str(prediction),

        "probability": probability

    }

