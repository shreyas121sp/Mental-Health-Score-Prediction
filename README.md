# Mental Health Score Prediction

Predicting a student's mental health score (0–10) from daily digital and lifestyle
habits — screen time, phone unlocks, sleep, study hours, physical activity and
stress level.

A scikit-learn Random Forest regression pipeline (**R² 0.88, MAE 0.35**) served
over a FastAPI endpoint, with a lightweight vanilla-JS frontend for instant
predictions.

> **Not a diagnostic tool.** This is a student data-science project. The score is
> a statistical estimate from survey data, not medical or psychological advice.

---

## Demo

| | |
|---|---|
| Live API | https://mansik-santulan-score.onrender.com |
| Endpoint | `POST /predict` |
| Frontend | open `index.html` in a browser |

---

## Dataset

`Student Social Media And Mental Health Impact.csv` — 5,000 student records,
12 features and one continuous target.

| Feature | Type | Notes |
|---|---|---|
| `Age` | numeric | 10–100 |
| `Gender` | nominal | Male / Female |
| `Country` | nominal | grouped to top 10 + `Other` |
| `Academic_Level` | nominal | High School / Undergraduate / Graduate |
| `Most_Used_Platform` | nominal | 12 platforms |
| `Purpose_Of_Use` | nominal | Networking / Education / Entertainment / News |
| `Avg_Daily_Usage_Hours` | numeric | hours |
| `Daily_Unlocks` | numeric | phone unlocks per day |
| `Study_Hours` | numeric | right-skewed → log-transformed |
| `Physical_Activity_Hours` | numeric | hours |
| `Sleep_Hours_Per_Night` | numeric | hours |
| `Stress_Level` | ordinal | Low < Medium < High < Very High |
| **`Mental_Health_Score`** | **target** | continuous, 0–10 |

---

## Modeling

Full walkthrough in [`ML_Project.ipynb`](ML_Project.ipynb) (EDA, feature
engineering, model comparison, tuning).

The preprocessing and the estimator live in a single `Pipeline`, so the exact
transformations used at training time are reapplied at inference — no train/serve
skew.

```
ColumnTransformer
├── Skewed    (Study_Hours)        → log1p → StandardScaler
├── Numeric   (Age, usage, unlocks, activity, sleep) → StandardScaler
├── Ordinal   (Stress_Level)       → OrdinalEncoder (Low→Very High)
└── Nominal   (Gender, Academic_Level, Platform, Purpose, Grouped_country)
                                   → OneHotEncoder
        ↓
RandomForestRegressor
```

High-cardinality `Country` is collapsed into `Grouped_country` — the 10 most
frequent countries, everything else as `Other` — to avoid a sparse one-hot blowup.
The API applies the same bucketing at request time.

### Results (30% hold-out, `random_state=42`)

| Model | Test R² | Train R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.740 | 0.724 | 0.536 | 0.676 |
| **Random Forest (default)** | **0.878** | 0.981 | **0.347** | **0.464** |
| Random Forest (tuned) | 0.865 | 0.955 | 0.369 | 0.487 |

`RandomizedSearchCV` (15 candidates, 5-fold, scoring `r2`) picked
`n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2`.
Tuning closed the train/test gap considerably but did not improve hold-out R²,
so the shipped `Mental_Health_Model.pkl` is the default Random Forest.

---

## Project structure

```
.
├── ML_Project.ipynb        # EDA, preprocessing, model training & evaluation
├── ML Project.html         # rendered notebook (no Jupyter needed)
├── Mental_Health_Model.pkl # trained sklearn Pipeline (~25 MB)
├── main.py                 # FastAPI service
├── index.html              # frontend form
├── script.js               # client-side validation + API call
├── style.css               # styling
├── requirements.txt
└── Student Social Media And Mental Health Impact.csv
```

---

## Running locally

```bash
git clone https://github.com/shreyas121sp/Mental-Health-Score-Prediction.git
cd Mental-Health-Score-Prediction

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

The API is then at `http://127.0.0.1:8000`, interactive docs at
`http://127.0.0.1:8000/docs`.

To point the frontend at your local server, edit `API_BASE` in
[`script.js`](script.js):

```js
const API_BASE = "http://127.0.0.1:8000";
```

---

## API

### `POST /predict`

**Request**

```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 5.5,
  "daily_unlocks": 120,
  "study_hours": 3.0,
  "physical_activity_hours": 1.0,
  "sleep_hours_per_night": 6.0,
  "stress_level": "High"
}
```

**Response**

```json
{ "predicted_mental_health_score": 5.42 }
```

**Validation** — enforced by Pydantic, returns `422` on violation:

| Field | Constraint |
|---|---|
| `age` | 10 – 100 |
| `avg_daily_usage_hours`, `study_hours`, `physical_activity_hours`, `sleep_hours_per_night` | 0 – 24 |
| `daily_unlocks` | ≥ 0 |
| `gender`, `academic_level`, `most_used_platform`, `purpose_of_use`, `stress_level` | fixed category sets |

**Try it**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":21,"gender":"Male","country":"India","academic_level":"Undergraduate","most_used_platform":"Instagram","purpose_of_use":"Entertainment","avg_daily_usage_hours":5.5,"daily_unlocks":120,"study_hours":3.0,"physical_activity_hours":1.0,"sleep_hours_per_night":6.0,"stress_level":"High"}'
```

---

## Tech stack

**Python** · scikit-learn · pandas · NumPy · FastAPI · Pydantic · Uvicorn ·
joblib · HTML/CSS/JavaScript

---

## Limitations

- Trained on self-reported survey data; correlation, not causation.
- `Gender` is limited to two categories, as in the source dataset.
- Predictions outside the training distribution (very rare countries, extreme
  usage values) are unreliable.
- The model has not been validated against any clinical instrument.
