# diagnosis/views.py

import pandas as pd
import pickle
# from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render, redirect


# Load models and encoders
with open('diagnosis\ml_models\label_encoders.pkl', 'rb') as f:
    le_animal, le_symptoms1, le_symptoms2, le_symptoms3, le_symptoms4, le_symptoms5, le_disease = pickle.load(f)

model = None

def load_model():
    global model
    if model is None:
        with open("diagnosis\ml_models\wrf_model.pkl", 'rb') as f:
            model = pickle.load(f)

def home(request):
    return render(request, 'index.html', {
        'animals': le_animal.classes_,
        'symptoms1': le_symptoms1.classes_,
        'symptoms2': le_symptoms2.classes_,
        'symptoms3': le_symptoms3.classes_,
        'symptoms4': le_symptoms4.classes_,
        'symptoms5': le_symptoms5.classes_,
    })

def predict(request):
    load_model()
    
    if request.method == 'POST':
        data = request.POST
        animal = data['animal']
        symptom1 = data['symptom1']
        symptom2 = data['symptom2']
        symptom3 = data['symptom3']
        symptom4 = data['symptom4']
        symptom5 = data['symptom5']

        # Encode the inputs
        animal_encoded = le_animal.transform([animal])[0]
        symptom1_encoded = le_symptoms1.transform([symptom1])[0]
        symptom2_encoded = le_symptoms2.transform([symptom2])[0]
        symptom3_encoded = le_symptoms3.transform([symptom3])[0]
        symptom4_encoded = le_symptoms4.transform([symptom4])[0]
        symptom5_encoded = le_symptoms5.transform([symptom5])[0]

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[animal_encoded, symptom1_encoded, symptom2_encoded, 
                                     symptom3_encoded, symptom4_encoded, symptom5_encoded]],
                                   columns=['Animal', 'Symptoms1', 'Symptoms2', 'Symptoms3', 'Symptoms4', 'Symptoms5'])

        # Make a prediction
        prediction = model.predict(input_data)[0]

        # Decode the predicted disease
        disease_decoded = le_disease.inverse_transform([prediction])[0]

        # return JsonResponse({'predicted_disease': disease_decoded})
        return render(request, 'result.html', {'predicted_disease': disease_decoded})

    return redirect('home')