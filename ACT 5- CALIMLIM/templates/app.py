import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the SVM model trained for credit card eligibility
model = pickle.load(open('models/model_act5.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # feature = request.form.values('spending')
    user_input = [float(request.form[field]) for field in ['age', 'experience', 'income', 'family', 'CCavg', 'education', 'mortgage', 'personal_loan', 'securities_account', 'cd_account', 'online']]
    user_input = np.array(user_input).reshape(1, -1)
    prediction = model.predict(user_input)
    print(prediction)
    result = 'Eligible for Credit Card' if prediction == 1 else 'Not Eligible for Credit Card'

    return render_template('index.html', prediction_output=f'Customer is {result}')


if __name__ == "__main__":
    app.run(debug=True)
