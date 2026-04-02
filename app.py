import gradio as gr

import pandas as pd
import pickle
import numpy as np

#load model
with open("random_forest_model.pkl","rb")as file:
    model=pickle.load(file)

def predict(Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,
       RestingECG,MaxHR,ExerciseAngina,Oldpeak, ST_Slope,HeartDisease):
    
    input_df = pd.DataFrame(
    [[
      Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,
       RestingECG,MaxHR,ExerciseAngina,Oldpeak, ST_Slope,HeartDisease
    ]],
    columns=[
       'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
       'HeartDisease'
    ]
)
    

    #prediction

    prediction=model.predict(input_df)[0]
    if prediction == 1:
        return "Result: Heart disease (Yes)"
    else:
        return "Result: Heart disease (No)"


inputs = [
    gr.Number(label="Age"),
    gr.Radio(choices=["M", "F"], label="Sex"),
    gr.Dropdown(choices=["ATA", "NAP", "ASY", "TA"], label="Chest Pain Type"),
    gr.Number(label="Resting Blood Pressure"),
    gr.Number(label="Cholesterol"),
    gr.Radio(choices=[0, 1], label="Fasting Blood Sugar"),
    gr.Dropdown(choices=["Normal", "ST", "LVH"], label="Resting ECG"),
    gr.Number(label="Max Heart Rate"),
    gr.Radio(choices=["Y", "N"], label="Exercise Angina"),
    gr.Number(label="Oldpeak"),
    gr.Dropdown(choices=["Up", "Flat", "Down"], label="ST Slope")
]  


app= gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Heart Disease Prediction"
)

app.launch(share=True)
