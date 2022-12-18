from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.compose import ColumnTransformer


app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get the form inputs
        rd_spend = float(request.form['rd_spend'])
        admin = float(request.form['admin'])
        marketing_spend = float(request.form['marketing_spend'])
        state = request.form['state']

        # Preprocess the inputs
        inputs = [[rd_spend, admin, marketing_spend, state, 0, 0]]
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot", OneHotEncoder(), [3])
            ],
            remainder='passthrough' 
        )
        inputs = transformer.fit_transform(inputs)

        # Load the trained model
        model = pickle.load(open('model.pkl', 'rb'))

        # Make a prediction
        prediction = model.predict(inputs)

        # Render the home page template and pass the prediction result
        return render_template('home.html', prediction= prediction[0])
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
