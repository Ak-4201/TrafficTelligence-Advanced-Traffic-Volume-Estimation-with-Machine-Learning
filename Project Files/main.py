import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

# Corrected app initialization
app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('G:/AI&ML/ML projects/Traffic_volume/model.pkl', 'rb'))
scale = pickle.load(open('C:/Users/SmartbridgePC/Desktop/AIML/Guided projects/scale.pkl', 'rb'))

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=["POST", "GET"])
def predict():
    try:
        # Read and convert inputs
        input_features = [float(x) for x in request.form.values()]
        feature_values = np.array(input_features).reshape(1, -1)

        
        column_names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year',
                        'month', 'day', 'hours', 'minutes', 'seconds']

        # Create DataFrame and scale it
        data = pd.DataFrame(feature_values, columns=column_names)
        data_scaled = scale.transform(data)

       
        prediction = model.predict(data_scaled)
        text = "Estimated Traffic Volume is: "
        return render_template("index.html", prediction_text=text + str(int(prediction[0])))

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)