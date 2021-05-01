from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('Pkl_Model.pkl', 'rb'))

@app.route('/')
@app.route('/index.html')
def home():
    return render_template("index.html")

@app.route('/formdata.html')
def form():
    return render_template("formdata.html")

@app.route('/result.html')
def result():
    return render_template("result.html")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['POST'])
def prediction():
    data1 = request.form['a']
    data2 = request.form['b']
    arr = np.array([[data1, data2]])
    # Predict the Model   
    pred = model.predict(arr)
    #print("Result: {0:.2f} %".format(100 * predict)) 
    return render_template('result.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)