import altair as alt
import streamlit as st
import pickle
import pandas as pd
import joblib

# Load the classifier from the pickle file
num_lab_procedures = st.number_input("Enter no. of lab procedures [1-132]:")
num_medications = st.number_input("Enter no. of medications [1-132]:")
time_in_hospital = st.number_input("Enter the amount of time in hospital [1-14]:")
number_inpatient = st.number_input("Enter the no. of inpatients [1-21]:")
number_diagnoses = st.number_input("Enter the no. of diagnoses [1-16]:")
discharge_disposition_id = st.number_input("Enter the discharge disposition ID [1-28]:")
admission_source_id = st.number_input("Enter the admission source ID [1-25]:")
number_outpatient = st.number_input("Enter the outpatient number [1-42]:")
number_emergency = st.number_input("Enter the emergency number [1-76]:")

def main():
    # Streamlit interface title
    st.title("Decision Tree Classifier Interface")
    # Perform classification based on user input
    if st.button("Classify"):
        # Create a DataFrame from user input
        user_data = pd.DataFrame(
            {"num_lab_procedures": [num_lab_procedures], 
             "num_medications": [num_medications], 
             "time_in_hospital": [time_in_hospital],
             "number_inpatient": [number_inpatient],
             "number_diagnoses": [number_diagnoses], 
             "discharge_disposition_id": [discharge_disposition_id], 
             "admission_source_id": [admission_source_id],
             "number_outpatient": [number_outpatient],
             "number_emergency": [number_emergency]         
            
            })
        from sklearn.preprocessing import StandardScaler

        # Create a StandardScaler instance
        scaler = StandardScaler()

        # Normalize the DataFrame
        normalized_data = pd.DataFrame(scaler.fit_transform(user_data), columns=user_data.columns)
        classifier = joblib.load('decision_tree_classifier.pkl')
        
        # Perform prediction using the loaded classifier
        prediction = classifier.predict(normalized_data)

        # Display the prediction
        st.subheader("Prediction")
        st.write(prediction)
        

# Run the Streamlit app
if __name__ == '__main__':
    main()
