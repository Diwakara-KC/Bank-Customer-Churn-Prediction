# Import Flask modules
from flask import Flask, render_template, request

# Feature Scaling
from sklearn.preprocessing import StandardScaler
import pickle

sc = pickle.load(open('standardscaler.pkl','rb'))
# Open our model
from tensorflow import keras
model = keras.models.load_model('ANN_model')


# Initialize Flask
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    geography = request.form['geography']
    creditscore = float(request.form['creditScore'])
    gender = float(request.form['gender'])
    age = float(request.form['age'])
    tenure = float(request.form['tenure'])
    balance = float(request.form['balance'])
    numofproducts = float(request.form['numofproducts'])
    hascrcard = float(request.form['hascrcard'])
    isactivemember = float(request.form['isactivemember'])
    estimatedsalary = float(request.form['estimatedsalary'])

    g1 = float(geography[0])
    g2 = float(geography[2])
    g3 = float(geography[4])


    pred = model.predict(sc.transform([[g1, g2, g3, creditscore, gender, age, tenure, balance, numofproducts, hascrcard, isactivemember, estimatedsalary]]))
    prediction = float(pred)


    if prediction >0.5:
        result = "Therefore, Our model predicts that the customer will not stay in the bank."
    else:
        result = "Therefore, Our model predicts that the customer will stay in the bank."


    return render_template('predict.html', prediction_text=result)


# Run app
if __name__ == "__main__":
    app.run(debug=True)
