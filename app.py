from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('diabetics_80.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def prediction():
    preg = int(request.form.get('preg'))
    plas = int(request.form.get('plas'))
    pres = int(request.form.get('pres'))
    skin = int(request.form.get('skin'))
    test = int(request.form.get('test'))
    mass = float(request.form.get('mass'))
    pedi = float(request.form.get('pedi'))
    age = int(request.form.get('age'))
    
    output = model.predict([[preg, plas, pres, skin, test, mass, pedi, age]])
    print(output)
    if output[0] == 0:
        data = "Person doesn't have diabetes"
    else:
        data = "Person has diabetes"
    print(data)

    return render_template('predict.html',data = data)
if __name__ == "__main__":
    app.run(debug=True)
