from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

application = Flask(__name__) 
app = application

# Reading pickles
random_forest_model = pickle.load(open('Models/random_forest_model.pkl', 'rb'))
standart_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

# Wine dataset min, max, and mean values
wine_data_stats = {
    'fixed_acidity': {'min': 4.6, 'max': 15.9, 'mean': 8.32},
    'volatile_acidity': {'min': 0.12, 'max': 1.58, 'mean': 0.53},
    'citric_acid': {'min': 0.0, 'max': 1.0, 'mean': 0.27},
    'residual_sugar': {'min': 0.9, 'max': 15.5, 'mean': 2.54},
    'chlorides': {'min': 0.012, 'max': 0.611, 'mean': 0.0875},
    'free_sulfur_dioxide': {'min': 1, 'max': 72, 'mean': 15.87},
    'total_sulfur_dioxide': {'min': 6, 'max': 289, 'mean': 46.47},
    'density': {'min': 0.99007, 'max': 1.00369, 'mean': 0.9967},
    'pH': {'min': 2.74, 'max': 4.01, 'mean': 3.31},
    'sulphates': {'min': 0.33, 'max': 2.0, 'mean': 0.658},
    'alcohol': {'min': 8.4, 'max': 14.9, 'mean': 10.42}
}

@app.route("/", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Get input values or use mean as default
            fixed_acidity = float(request.form.get('fixed_acidity', wine_data_stats['fixed_acidity']['mean']))
            volatile_acidity = float(request.form.get('volatile_acidity', wine_data_stats['volatile_acidity']['mean']))
            citric_acid = float(request.form.get('citric_acid', wine_data_stats['citric_acid']['mean']))
            residual_sugar = float(request.form.get('residual_sugar', wine_data_stats['residual_sugar']['mean']))
            chlorides = float(request.form.get('chlorides', wine_data_stats['chlorides']['mean']))
            free_sulfur_dioxide = float(request.form.get('free_sulfur_dioxide', wine_data_stats['free_sulfur_dioxide']['mean']))
            total_sulfur_dioxide = float(request.form.get('total_sulfur_dioxide', wine_data_stats['total_sulfur_dioxide']['mean']))
            density = float(request.form.get('density', wine_data_stats['density']['mean']))
            pH = float(request.form.get('pH', wine_data_stats['pH']['mean']))
            sulphates = float(request.form.get('sulphates', wine_data_stats['sulphates']['mean']))
            alcohol = float(request.form.get('alcohol', wine_data_stats['alcohol']['mean']))
        except ValueError as e:
            return render_template('home.html', result_value=None, level=f"Input error: {e}")

        # Scaling and predicting
        new_data_scaled = standart_scaler.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                                      chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                                                      pH, sulphates, alcohol]])
        result = random_forest_model.predict(new_data_scaled)
        result_value = result[0]
        print(result_value)
        # Quality message
        if result_value == 1:
            txt = 'Хороша якість вина'
        else:
            txt = 'Погана якість вина'

        # Rendering result in the template, passing wine_data_stats as well
        return render_template('home.html', result_value=txt, quality=result_value, 
                               fixed_acidity=fixed_acidity, volatile_acidity=volatile_acidity, citric_acid=citric_acid, 
                               residual_sugar=residual_sugar, chlorides=chlorides, free_sulfur_dioxide=free_sulfur_dioxide, 
                               total_sulfur_dioxide=total_sulfur_dioxide, density=density, pH=pH, sulphates=sulphates, alcohol=alcohol,
                               wine_data_stats=wine_data_stats)

    return render_template('home.html', result_value=None, quality=None, wine_data_stats=wine_data_stats)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
