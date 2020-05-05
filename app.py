import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
drugmodel = pickle.load(open('drugmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = drugmodel.predict(final_features)

    output = round(prediction[0], 2)
    
    if output == 1:
        return render_template('index.html', prediction_text='Addicted Possibility: YES')
    else:
        return render_template('index.html', prediction_text='Addicted Possibility: NO')
    

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = drugmodel.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)