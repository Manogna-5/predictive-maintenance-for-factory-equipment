import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Globals to store model and data
model = None
accuracy = None
uploaded_data = None
scaler = None  # For scaling when needed

# Feature columns
FEATURE_COLUMNS = ['temperature', 'vibration', 'pressure', 'rpm', 'load', 'runtime', 'humidity', 'last_maintenance']

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, accuracy, uploaded_data, scaler

    prediction = None
    classifier_name = request.form.get('classifier') if request.method == 'POST' else 'Random Forest'

    if request.method == 'POST':
        # Handle CSV upload
        if 'dataset' in request.files:
            file = request.files['dataset']
            if file:
                path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(path)
                uploaded_data = pd.read_csv(path)
                print("✅ Uploaded CSV loaded successfully")
                print("📊 Failure column distribution:")
                print(uploaded_data['failure'].value_counts())  # Show class balance

        # Train Model
        elif request.form.get('action') == 'train' and uploaded_data is not None:
            X = uploaded_data[FEATURE_COLUMNS]
            y = uploaded_data['failure']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            try:
                if classifier_name in ['SVM', 'Logistic Regression']:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                else:
                    scaler = None  # Not needed

                if classifier_name == 'Random Forest':
                    model = RandomForestClassifier(class_weight='balanced')
                elif classifier_name == 'Logistic Regression':
                    model = LogisticRegression(max_iter=1000, class_weight='balanced')
                elif classifier_name == 'SVM':
                    model = SVC(class_weight='balanced')
                else:
                    model = RandomForestClassifier(class_weight='balanced')

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred) * 100
                print("✅ Model trained successfully")
                print("🧠 Predictions in test set:", set(y_pred))  # Show what model is predicting

            except Exception as e:
                accuracy = "Training Error"
                print(f"❌ Error during model training: {e}")

        # Prediction
        elif all(k in request.form for k in FEATURE_COLUMNS) and model:
            try:
                values = [float(request.form[col]) for col in FEATURE_COLUMNS]
                pred_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)

                if classifier_name in ['SVM', 'Logistic Regression'] and scaler:
                    pred_df = scaler.transform(pred_df)

                result = model.predict(pred_df)[0]

                # ✅ Map prediction to readable string
                prediction = "Failure" if result == 1 else "No Failure"
                print(f"🔍 Predicted value: {result} → {prediction}")
            except ValueError:
                prediction = 'Invalid input'
            except Exception as e:
                prediction = f'Prediction error: {e}'
                print(f"❌ Error during prediction: {e}")

    return render_template('index.html',
                           data=uploaded_data.to_html(classes='table', index=False) if uploaded_data is not None else None,
                           accuracy=accuracy,
                           prediction=prediction,
                           classifier=classifier_name)

# ✅ Render-compatible entrypoint
if _name_ == '_main_':
    os.makedirs('uploads', exist_ok=True)
    port = int(os.environ.get("PORT", 5000))  # For Render/Heroku
    app.run(host='0.0.0.0', port=port, debug=True)