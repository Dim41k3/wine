from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

application = Flask(__name__) 
app = application

random_forest_model = pickle.load(open('Models/best_wine_quality_model.pkl', 'rb'))
standart_scaler = pickle.load(open('Models/wine_quality_scaler.pkl', 'rb'))

@app.route("/", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            fixed_acidity = float(request.form.get('fixed_acidity', 8.32))
            volatile_acidity = float(request.form.get('volatile_acidity', 0.53))
            citric_acid = float(request.form.get('citric_acid', 0.27))
            residual_sugar = float(request.form.get('residual_sugar', 2.54))
            chlorides = float(request.form.get('chlorides', 0.0875))
            free_sulfur_dioxide = float(request.form.get('free_sulfur_dioxide', 15.87))
            total_sulfur_dioxide = float(request.form.get('total_sulfur_dioxide', 46.47))
            density = float(request.form.get('density', 0.9967))
            pH = float(request.form.get('pH', 3.31))
            sulphates = float(request.form.get('sulphates', 0.658))
            alcohol = float(request.form.get('alcohol', 10.42))
        except ValueError as e:
            return render_template('home.html', result_value=None, level=f"Input error: {e}")

        new_data_scaled = standart_scaler.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                                      chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                                                      pH, sulphates, alcohol]])
        result = random_forest_model.predict(new_data_scaled)
        result_value = result[0]
        print(result_value)

        return render_template('home.html', quality=result_value, 
                               fixed_acidity=fixed_acidity, volatile_acidity=volatile_acidity, citric_acid=citric_acid, 
                               residual_sugar=residual_sugar, chlorides=chlorides, free_sulfur_dioxide=free_sulfur_dioxide, 
                               total_sulfur_dioxide=total_sulfur_dioxide, density=density, pH=pH, sulphates=sulphates, alcohol=alcohol)

    return render_template('home.html', result_value=None, quality=None)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
