import gradio as gr
import joblib
import numpy as np

# Load models
knn_model = joblib.load("knn_model.pkl")
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define prediction function
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                     bmi, diabetes_pedigree_function, age, model_choice):

    # Combine inputs into array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                            bmi, diabetes_pedigree_function, age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Select model
    if model_choice == "KNN":
        model = knn_model
    else:
        model = svm_model

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence = f"{prob * 100:.2f}%"

    return result, confidence

# UI Inputs
input_fields = [
    gr.Slider(0, 20, step=1, label="Pregnancies"),
    gr.Slider(0, 200, step=1, label="Glucose"),
    gr.Slider(0, 130, step=1, label="Blood Pressure"),
    gr.Slider(0, 100, step=1, label="Skin Thickness"),
    gr.Slider(0, 846, step=1, label="Insulin"),
    gr.Slider(0.0, 70.0, step=0.1, label="BMI"),
    gr.Slider(0.0, 2.5, step=0.01, label="Diabetes Pedigree Function"),
    gr.Slider(10, 100, step=1, label="Age"),
    gr.Radio(["KNN", "SVM"], label="Select Model")
]

# UI Output
output_fields = [
    gr.Textbox(label="Prediction"),
    gr.Textbox(label="Confidence")
]

# Launch interface
gr.Interface(
    fn=predict_diabetes,
    inputs=input_fields,
    outputs=output_fields,
    title="Diabetes Prediction App",
    description="Enter patient details and choose a model (KNN or SVM) to predict diabetes."
).launch()
