import os
import io
import pandas as pd
from flask import Flask, render_template, request, send_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Folder for uploaded CSV files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

FEATURE_COLUMNS = ['temperature', 'vibration', 'pressure', 'rpm', 'load', 'runtime', 'humidity', 'last_maintenance']

# Variables to hold the model and other details
model = None
accuracy = None
uploaded_data = None
scaler = None
last_trained_classifier = None
prediction_log = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def index():
    global model, accuracy, uploaded_data, scaler, last_trained_classifier, prediction_log

    prediction = None
    classifier_name = request.form.get('classifier') or last_trained_classifier or 'Random Forest'

    if request.method == 'POST':
        # 1. Handle CSV Upload
        if 'dataset' in request.files:
            file = request.files['dataset']
            if file:
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(path)
                uploaded_data = pd.read_csv(path)
                prediction_log = []
                print("✅ CSV loaded")

        # 2. Train Model
        elif request.form.get('action') == 'train' and uploaded_data is not None:
            X = uploaded_data[FEATURE_COLUMNS]
            y = uploaded_data['failure']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            try:
                if classifier_name in ['SVM', 'Logistic Regression', 'KNN']:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                else:
                    scaler = None

                if classifier_name == 'Random Forest':
                    model = RandomForestClassifier(class_weight='balanced')
                elif classifier_name == 'Logistic Regression':
                    model = LogisticRegression(max_iter=1000, class_weight='balanced')
                elif classifier_name == 'SVM':
                    model = SVC(class_weight='balanced')
                elif classifier_name == 'KNN':
                    model = KNeighborsClassifier()
                else:
                    model = RandomForestClassifier(class_weight='balanced')

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
                last_trained_classifier = classifier_name
                print(f"✅ {classifier_name} trained")

            except Exception as e:
                accuracy = "Training Error"
                print(f"❌ Training error: {e}")

        # 3. Prediction
        elif all(k in request.form for k in FEATURE_COLUMNS) and model:
            try:
                values = [float(request.form[col]) for col in FEATURE_COLUMNS]
                pred_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
                if last_trained_classifier in ['SVM', 'Logistic Regression', 'KNN'] and scaler:
                    pred_df = scaler.transform(pred_df)
                result = model.predict(pred_df)[0]
                prediction = result

                prediction_log.append({**dict(zip(FEATURE_COLUMNS, values)), "prediction": int(result)})
            except Exception as e:
                prediction = f'Prediction error: {e}'

    failure_count = None
    if prediction_log:
        failure_count = {"0": 0, "1": 0}
        for entry in prediction_log:
            pred_val = entry.get("prediction")
            if pred_val == 1:
                failure_count["1"] += 1
            else:
                failure_count["0"] += 1

    return render_template('index.html',
                           data=uploaded_data.to_html(classes='table', index=False) if uploaded_data is not None else None,
                           accuracy=accuracy,
                           prediction=prediction,
                           classifier=last_trained_classifier or 'Random Forest',
                           prediction_log=prediction_log,
                           failure_count=failure_count)

@app.route('/download_predictions')
def download_predictions():
    if not prediction_log:
        return "No predictions yet."
    df = pd.DataFrame(prediction_log)
    csv = df.to_csv(index=False)
    return send_file(io.BytesIO(csv.encode('utf-8')),
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name='predictions.csv')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)