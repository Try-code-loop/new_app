import streamlit as st 
import pandas as pd
import altair as alt
from predict_function import predict

# Apply light blue background using custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6; /* Light blue background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Health Screening")
st.subheader("for Coronary Heart Disease or Myocardial Infarction")

# Sidebar Notes
st.sidebar.title("Notes")
st.sidebar.subheader("Project")
st.sidebar.write("""
This is primarily a data science project that utilized health data. By no means it can replace a doctor, nurse, or other health professional.
The application raises awareness of heart disease risk factors. Heart diseases are a leading cause of morbidity and mortality worldwide. 
A better understanding of the contributing factors can improve early interventions.
""")
st.sidebar.subheader("Model and Data")
st.sidebar.write("""
The user input is evaluated with a pre-trained machine learning model (random forest). 
The model was trained on a dataset containing 253,680 survey responses from cleaned 
Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey data.
""")
st.sidebar.subheader("Model Evaluation")
st.sidebar.write("""
The model is more sensitive towards risk factors for heart diseases as they become more 
significant (e.g. very high blood pressure or high cholesterol) and/or with the accumulation 
of risk factors (e.g. low general or mental health and more). 

As a result, the model may be biased towards predicting higher chances of heart diseases, 
while the precision of this prediction may be low. However, in this regard, it captures 
most actual heart disease cases (high recall rate). This outcome is assumed to be suitable, 
given the aim of the application to sensitize the user to the risks of having a heart disease.
""")

# Initial example_input dictionary for default values
example_input = {
    'HighBP': 1,
    'HighChol': 0,
    'CholCheck': 0,
    'BMI': 25,
    'Smoker': 1,
    'Stroke': 0,
    'Diabetes': 0,
    'PhysActivity': 1,
    'Fruits': 1,
    'Veggies': 0,
    'HvyAlcoholConsump': 0,
    'AnyHealthcare': 1,
    'NoDocbcCost': 0,
    'GenHlth': 3,
    'MentHlth': 0,
    'PhysHlth': 0,
    'DiffWalk': 0,
    'Sex': 0,
    'Age': 7,
    'Education': 3,
    'Income': 4
}

# Age group mapping
age_groups = {
    "Age 18 to 24": 1,
    "Age 25 to 29": 2,
    "Age 30 to 34": 3,
    "Age 35 to 39": 4,
    "Age 40 to 44": 5,
    "Age 45 to 49": 6,
    "Age 50 to 54": 7,
    "Age 55 to 59": 8,
    "Age 60 to 64": 9,
    "Age 65 to 69": 10,
    "Age 70 to 74": 11,
    "Age 75 to 79": 12,
    "Age 80 or older": 13
}

# Education mapping
education_levels = {
    "Never attended school or only kindergarten": 1,
    "Grades 1 through 8 (Elementary)": 2,
    "Grades 9 through 11 (Some high school)": 3,
    "Grade 12 or GED (High school graduate)": 4,
    "College 1 year to 3 years (Some college or technical school)": 5,
    "College 4 years or more (College graduate)": 6
}

# Income mapping
income_levels = {
    "Less than $10,000": 1,
    "Less than $15,000 ($10,000 to less than $15,000)": 2,
    "Less than $20,000 ($15,000 to less than $20,000)": 3,
    "Less than $25,000 ($20,000 to less than $25,000)": 4,
    "Less than $35,000 ($25,000 to less than $35,000)": 5,
    "Less than $50,000 ($35,000 to less than $50,000)": 6,
    "Less than $75,000 ($50,000 to less than $75,000)": 7,
    "$75,000 or more": 8
}

# General health mapping
gen_health_levels = {
    "excellent": 1,
    "very good": 2,
    "good": 3,
    "fair": 4,
    "poor": 5
}

# Reverse mapping for defaults
age_group_labels = list(age_groups.keys())
default_age_label = age_group_labels[example_input['Age'] - 1]

education_labels = list(education_levels.keys())
default_education_label = education_labels[example_input['Education'] - 1]

income_labels = list(income_levels.keys())
default_income_label = income_labels[example_input['Income'] - 1]

gen_health_labels = list(gen_health_levels.keys())
default_gen_health_label = gen_health_labels[example_input['GenHlth'] - 1]

# Input fields
with st.form("health_form"):
    HighBP = st.selectbox('Has High Blood Pressure been confirmed by a doctor, nurse, or other health professional?', [0, 1], index=example_input['HighBP'])
    HighChol = st.selectbox('Have you ever been told by a doctor, nurse or other health professional that your blood cholesterol is high?', [0, 1], index=example_input['HighChol'])
    CholCheck = st.selectbox('Did you have a cholesterol check within the past five years', [0, 1], index=example_input['CholCheck'])
    BMIndex = st.number_input('What is your Body Mass Index (BMI)', min_value=10, max_value=30, value=int(example_input['BMI']))
    Smoker = st.selectbox('Have you smoked at least 100 cigarettes in your entire life?', [0, 1], index=example_input['Smoker'])
    Stroke = st.selectbox('Did you ever had a stroke?', [0, 1], index=example_input['Stroke'])
    Diabetes = st.selectbox('Do you have diabetes? If yes, dou you have diabetes level 1 or 2?', [0, 1, 2], index=example_input['Diabetes'])
    PhysActivity = st.selectbox('Did you do physical activity or exercise during the past 30 days?', [0, 1], index=example_input['PhysActivity'])
    Fruits = st.selectbox('Do you consume fruits one or more times per day?', [0, 1], index=example_input['Fruits'])
    Veggies = st.selectbox('Do you consume vegetables one or more times per day?', [0, 1], index=example_input['Veggies'])
    HvyAlcoholConsump = st.selectbox('Do you have more than 14 (men) / 7 (women) drinks per week?', [0, 1], index=example_input['HvyAlcoholConsump'])
    AnyHealthcare = st.selectbox('Do you have any kind of health care coverage?', [0, 1], index=example_input['AnyHealthcare'])
    NoDocbcCost = st.selectbox('Was there a time in the past 12 months when you needed to see a doctor but could not?', [0, 1], index=example_input['NoDocbcCost'])
    GenHlth = st.selectbox('How would you describe your general health?', gen_health_labels, index=example_input['GenHlth'] - 1)
    MentHlth = st.number_input('Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?', min_value=0, max_value=30, value=int(example_input['MentHlth']))
    PhysHlth = st.number_input('Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?', min_value=0, max_value=30, value=int(example_input['PhysHlth']))
    DiffWalk = st.selectbox('Do you have serious difficulty walking or climbing stairs?', [0, 1], index=example_input['DiffWalk'])
    Sex = st.selectbox('Sex (0=Female, 1=Male)', [0, 1], index=example_input['Sex'])
    Age = st.selectbox('Select your Age Group:', age_group_labels, index=example_input['Age'] - 1)
    Education = st.selectbox('What is your highest level of education?', education_labels, index=example_input['Education'] - 1)
    Income = st.selectbox('What is your annual income?', income_labels, index=example_input['Income'] - 1)

    # Confirm button
    submitted = st.form_submit_button("Confirm")

# If the confirm button is clicked, update example_input and display the prediction
if submitted:
    # Update the example_input dictionary with new values
    example_input.update({
        'HighBP': HighBP,
        'HighChol': HighChol,
        'CholCheck': CholCheck,
        'BMI': BMIndex,
        'Smoker': Smoker,
        'Stroke': Stroke,
        'Diabetes': Diabetes,
        'PhysActivity': PhysActivity,
        'Fruits': Fruits,
        'Veggies': Veggies,
        'HvyAlcoholConsump': HvyAlcoholConsump,
        'AnyHealthcare': AnyHealthcare,
        'NoDocbcCost': NoDocbcCost,
        'GenHlth': gen_health_levels[GenHlth],  # Map selected label to its corresponding value
        'MentHlth': MentHlth,
        'PhysHlth': PhysHlth,
        'DiffWalk': DiffWalk,
        'Sex': Sex,
        'Age': age_groups[Age],  # Map selected label to its corresponding value
        'Education': education_levels[Education],  # Map selected label to its corresponding value
        'Income': income_levels[Income]  # Map selected label to its corresponding value
    })

    # Predict function call with updated example_input
    example_output = predict(example_input)

    # Create main analysis DataFrame
    #st.write("Estimation:", example_output)

    # Display the prediction
    st.write(f"The risk for a heart disease is currently **'{example_output['prediction']}'**")

    # Display the probability as a percentage
    heart_disease_risk = example_output["probability"] * 100
    st.write(f"Heart Disease Risk in Percentage: **{heart_disease_risk:.2f}%**")

    # Handle the top_factors data
    top_factors_df = example_output["top_factors"]

    # Sort by contribution and take the top 5
    top_factors_df = top_factors_df.sort_values(by="contribution", ascending=False).head(5)

    # Display a chart for the top 5 factors
    st.write("### Top 5 Contributing Factors")
    chart = alt.Chart(top_factors_df).mark_bar().encode(
        x=alt.X('contribution:Q', title="Contribution to Risk"),
        y=alt.Y('feature:N', sort='-x', title="Feature"),
        color=alt.Color('feature:N', legend=None)
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart)
        
    # Ensure 'top_factors' DataFrame exists and extract the top 5
    if 'top_factors' in example_output:
        top_factors_df = pd.DataFrame(example_output['top_factors']).head(5)
        top_factors = top_factors_df['feature'].tolist()  # Get top 5 features as a list
    else:
        top_factors = []  # Default to an empty list if 'top_factors' is missing

    # Advice dictionary
    advice = {
        'HighBP': "Maintain a healthy weight, reduce sodium intake, engage in regular exercise, and manage stress to lower blood pressure.",
        'HighChol': "Consume a heart-healthy diet with less saturated fat, avoid trans fats, and include foods rich in omega-3 fatty acids.",
        'CholCheck': "Schedule regular cholesterol screenings every 4-6 years or as recommended by your doctor.",
        'BMI': "Aim for a balanced diet and consistent physical activity to maintain a BMI within the normal range (18.5–24.9).",
        'Smoker': "Consider smoking cessation programs, nicotine replacement therapy, or behavioral support to quit smoking.",
        'Stroke': "Control high blood pressure, manage diabetes, stay physically active, and avoid smoking or heavy drinking.",
        'Diabetes': "Maintain blood sugar levels through a healthy diet, regular exercise, and prescribed medication or insulin therapy.",
        'PhysActivity': "Incorporate at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week.",
        'Fruits': "Include at least one serving of fruits daily as part of a balanced diet.",
        'Veggies': "Consume a variety of vegetables daily, aiming for at least 2-3 servings.",
        'HvyAlcoholConsump': "Limit alcohol intake to no more than 14 drinks per week for men or 7 drinks per week for women.",
        'AnyHealthcare': "Ensure consistent access to healthcare coverage to address any potential health issues promptly.",
        'NoDocbcCost': "Explore community health resources or sliding-scale clinics for affordable medical care.",
        'GenHlth': "Focus on preventive care, regular check-ups, and a balanced lifestyle to improve general health.",
        'MentHlth': "Seek support for mental health challenges through therapy, counseling, or stress-reduction techniques.",
        'PhysHlth': "Address chronic pain or illnesses through regular medical consultations and personalized care plans.",
        'DiffWalk': "Engage in physical therapy or use assistive devices to improve mobility and reduce discomfort.",
        'Sex': "Focus on gender-specific health screenings (e.g., mammograms, prostate exams) and preventive care.",
        'Age': "Follow age-appropriate health guidelines, including screenings and vaccinations.",
        'Education': "Leverage educational resources to improve health literacy and awareness about healthy lifestyle choices.",
        'Income': "Utilize community resources and budgeting strategies to access affordable healthcare and nutritious food."
    }

    # Display advice for only top 5 factors
    st.write("### Personalized Health Advice for Top Risk Factors")
    for factor in top_factors:
        if factor in advice:
            st.write(f"**{factor}:** {advice[factor]}")
