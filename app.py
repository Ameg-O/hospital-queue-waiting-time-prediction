"""
Hospital Queue Waiting-Time Prediction - Gradio App


Run:
    python app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import gradio as gr

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------- 1. Load and prepare dataset --------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "hospital_queue_prediction_advanced_with_nulls_50000.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place the CSV file in the same folder as app.py."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "waiting_time_minutes", "patient_id", "consultation_start_datetime",
    "arrival_datetime", "priority_label", "patient_race", "patient_gender",
    "admission_required", "triage_level", "queue_length_at_arrival",
    "available_doctors", "doctors_currently_busy", "staff_on_duty"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Use the same feature engineering steps as the notebook.
df["arrival_datetime"] = pd.to_datetime(df["arrival_datetime"])
df["arrival_hour"] = df["arrival_datetime"].dt.hour
df["arrival_day"] = df["arrival_datetime"].dt.day
df["arrival_dayofweek"] = df["arrival_datetime"].dt.dayofweek
df["is_weekend"] = (df["arrival_dayofweek"] >= 5).astype(int)

df["queue_per_doctor"] = (
    df["queue_length_at_arrival"] /
    df["available_doctors"].clip(lower=1)
)
df["busy_doctor_ratio"] = (
    df["doctors_currently_busy"] /
    df["staff_on_duty"].clip(lower=1)
)

df = df.sort_values("arrival_datetime").reset_index(drop=True)

# Same target and excluded columns as the notebook.
x = df.drop(columns=[
    "waiting_time_minutes",
    "patient_id",
    "consultation_start_datetime",
    "arrival_datetime",
    "priority_label",
    "patient_race",
    "patient_gender",
    "admission_required"
])
y = df["waiting_time_minutes"]

# Same chronological 80/20 train-test split.
split_point = int(len(df) * 0.80)
xtrain, xtest = x.iloc[:split_point], x.iloc[split_point:]
ytrain, ytest = y.iloc[:split_point], y.iloc[split_point:]

# Same numerical and categorical preprocessing approach.
numeric_features = xtrain.select_dtypes(include=np.number).columns
categorical_features = xtrain.select_dtypes(include=["object", "category"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Same tuned Random Forest search configuration as the notebook.
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
])
parameters = {
    "regressor__n_estimators": [100, 150],
    "regressor__max_depth": [10, 15, 20],
    "regressor__min_samples_leaf": [1, 2]
}
time_cv = TimeSeriesSplit(n_splits=3)
grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=parameters,
    scoring="neg_mean_absolute_error",
    cv=time_cv,
    n_jobs=-1,
    verbose=1
)

print("Training the tuned Random Forest model. This may take some time...")
grid.fit(xtrain, ytrain)
best_model = grid.best_estimator_

test_predictions = best_model.predict(xtest)
model_mae = mean_absolute_error(ytest, test_predictions)
model_rmse = np.sqrt(mean_squared_error(ytest, test_predictions))
model_r2 = r2_score(ytest, test_predictions)

print("Model training completed.")
print("Best parameters:", grid.best_params_)


# -------------------- 2. Prediction function --------------------

def predict_waiting_time(
    department,
    service_type,
    patient_age,
    triage_level,
    hour_of_day,
    queue_length_at_arrival,
    available_doctors,
    staff_on_duty,
    patients_arrived_last_30min,
    average_service_time_min,
    doctors_currently_busy,
    hospital_load_level,
    arrival_date
):
    try:
        # The notebook's feature set includes both day/hour fields and
        # date-derived fields. The date input supplies arrival_day and
        # arrival_dayofweek; hour_of_day supplies arrival_hour.
        arrival_dt = pd.to_datetime(arrival_date, errors="raise")

        patient = pd.DataFrame([{
            "department": str(department).strip(),
            "service_type": str(service_type).strip(),
            "patient_age": float(patient_age),
            "triage_level": float(triage_level),
            "day_of_week": arrival_dt.strftime("%A"),
            "hour_of_day": int(hour_of_day),
            "is_weekend": int(arrival_dt.dayofweek >= 5),
            "queue_length_at_arrival": float(queue_length_at_arrival),
            "available_doctors": float(available_doctors),
            "staff_on_duty": float(staff_on_duty),
            "patients_arrived_last_30min": float(patients_arrived_last_30min),
            "average_service_time_min": float(average_service_time_min),
            "doctors_currently_busy": float(doctors_currently_busy),
            "hospital_load_level": str(hospital_load_level),
            "arrival_hour": int(hour_of_day),
            "arrival_day": int(arrival_dt.day),
            "arrival_dayofweek": int(arrival_dt.dayofweek),
            "queue_per_doctor": float(queue_length_at_arrival) / max(float(available_doctors), 1),
            "busy_doctor_ratio": float(doctors_currently_busy) / max(float(staff_on_duty), 1)
        }])

        # Align columns to the exact training feature order.
        patient = patient.reindex(columns=xtrain.columns)

        prediction = float(best_model.predict(patient)[0])
        return (
            f"### Estimated waiting time: {prediction:.1f} minutes\n"
            f"Approximately **{prediction / 60:.2f} hours**."
        )
    except Exception as exc:
        return f"Could not make a prediction. Please check the inputs.\n\nError: {exc}"


# -------------------- 3. Gradio interface --------------------


with gr.Blocks(title="Hospital Queue Waiting-Time Predictor") as demo:
    gr.Markdown(
        "# 🏥 Hospital Queue Waiting-Time Prediction\n"
        "Enter the patient's and hospital's current conditions to estimate "
        "the patient's waiting time."
    )

    # Build dropdown choices directly from the dataset so the options
    # always match the categories used during model training.
    def dropdown_choices(column, fallback):
        if column in df.columns:
            values = df[column].dropna().astype(str).unique().tolist()
            values = sorted(values)
            if values:
                return values
        return fallback

    department_choices = dropdown_choices("department", ["Cardiology"])
    service_type_choices = dropdown_choices("service_type", ["Consultation"])
    load_level_choices = dropdown_choices(
        "hospital_load_level", ["Low", "Medium", "High", "Critical"]
    )

    with gr.Row():
        with gr.Column():
            department = gr.Dropdown(
                choices=department_choices,
                label="Department",
                value=department_choices[0]
            )
            service_type = gr.Dropdown(
                choices=service_type_choices,
                label="Service type",
                value=service_type_choices[0]
            )
            patient_age = gr.Number(label="Patient age", value=45, minimum=0, maximum=120)
            triage_level = gr.Number(label="Triage level (1–5)", value=3, minimum=1, maximum=5)
            arrival_date = gr.Textbox(
                label="Arrival date (YYYY-MM-DD)",
                value="2026-11-25",
                placeholder="YYYY-MM-DD"
            )
            hour_of_day = gr.Slider(0, 23, value=10, step=1, label="Arrival hour (0–23)")
        with gr.Column():
            queue_length_at_arrival = gr.Number(label="Queue length at arrival", value=15, minimum=0)
            available_doctors = gr.Number(label="Available doctors", value=5, minimum=0)
            staff_on_duty = gr.Number(label="Staff on duty", value=8, minimum=0)
            patients_arrived_last_30min = gr.Number(
                label="Patients arrived in last 30 minutes", value=10, minimum=0
            )
            average_service_time_min = gr.Number(
                label="Average service time (minutes)", value=15, minimum=0
            )
            doctors_currently_busy = gr.Number(
                label="Doctors currently busy", value=3, minimum=0
            )
            hospital_load_level = gr.Dropdown(
                choices=load_level_choices,
                label="Hospital load level",
                value=load_level_choices[0]
            )

    predict_button = gr.Button("Predict Waiting Time", variant="primary")
    output = gr.Markdown(label="Prediction")

    predict_button.click(
        fn=predict_waiting_time,
        inputs=[
            department, service_type, patient_age, triage_level,
            hour_of_day, queue_length_at_arrival, available_doctors,
            staff_on_duty, patients_arrived_last_30min, average_service_time_min,
            doctors_currently_busy, hospital_load_level, arrival_date
        ],
        outputs=output
    )

    gr.Markdown(
        "> **Note:** This is a machine-learning estimate based on the supplied "
        "dataset. It is not a guaranteed waiting time and should not be used "
        "as the only basis for clinical or operational decisions."
    )

if __name__ == "__main__":
    demo.launch()
