import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Globals to store model and data
model = None
accuracy = None
uploaded_data = None

# Feature columns
FEATURE_COLUMNS = ['temperature', 'vibration', 'pressure', 'rpm', 'load', 'runtime', 'humidity', 'last_maintenance']

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, accuracy, uploaded_data

    prediction = None
    classifier_name = request.form.get('classifier') if request.method == 'POST' else 'Random Forest'

    if request.method == 'POST':
        # Load dataset
        if 'dataset' in request.files:
            file = request.files['dataset']
            if file:
                path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(path)
                uploaded_data = pd.read_csv(path)

        # Train Model
        elif request.form.get('action') == 'train' and uploaded_data is not None:
            X = uploaded_data[FEATURE_COLUMNS]
            y = uploaded_data['failure']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if classifier_name == 'Random Forest':
                model = RandomForestClassifier()
            elif classifier_name == 'Logistic Regression':
                model = LogisticRegression(max_iter=1000)
            elif classifier_name == 'SVM':
                model = SVC()
            else:
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred) * 100

        # Predict
        elif request.form.get('action') == 'predict' and model:
            try:
                values = [float(request.form[col]) for col in FEATURE_COLUMNS]
                pred_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
                prediction = model.predict(pred_df)[0]
            except ValueError:
                prediction = 'Invalid input'

    return render_template('index.html',
                           data=uploaded_data.to_html(classes='table', index=False) if uploaded_data is not None else None,
                           accuracy=accuracy,
                           prediction=prediction,
                           classifier=classifier_name)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)