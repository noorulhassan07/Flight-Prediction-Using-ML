from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/flight_model.pkl")
encoders = joblib.load("model/encoders.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    carrier = encoders['OP_UNIQUE_CARRIER'].transform([request.form['carrier']])[0]
    origin = encoders['ORIGIN'].transform([request.form['origin']])[0]
    dest = encoders['DEST'].transform([request.form['dest']])[0]
    dep_time = int(request.form['dep_time'])

    input_data = np.array([[carrier, origin, dest, dep_time]])
    prediction = model.predict(input_data)[0]
    result = "✈ Flight is Delayed" if prediction == 1 else "🟢 Flight is On Time"
    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
