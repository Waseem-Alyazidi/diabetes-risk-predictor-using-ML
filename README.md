# Diabetes Risk Prediction Web Application

## Overview
This repository hosts a **full-stack web application** that predicts the risk of developing diabetes within five years based on simple health and lifestyle inputs. The project demonstrates a complete pipeline:

- Data utilities
- A manually implemented Random Forest classifier (no external ML libraries for model internals)
- A Flask backend exposing a prediction endpoint
- A responsive frontend for interactive user input

This project is designed for **educational purposes** and as a **proof of concept** for end-to-end machine learning applications.

---

## Key Features
- **End-to-end implementation**: From data loading and preprocessing to model training, inference, and a web interface.  
- **Custom ML implementation**: Decision Tree and Random Forest algorithms built from scratch using NumPy.  
- **Backend API**: Flask application exposing a `/predict` endpoint for programmatic access.  
- **Frontend**: Lightweight, responsive HTML/CSS/JS interface for interactive predictions.  
- **Reproducible training**: Jupyter notebook for training and serializing the model using `dill`.

---

## Repository Structure
```

Diabetes Risk Predictor/
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── model.pkl
│   └── utils.py
├── data/
│   └── diabetes.csv
├── frontend/
│   ├── statics/
│   │   ├── script.js
│   │   └── style.css
│   └── templates/
│       └── index.html
├── notebook/
│   └── train_model.ipynb
├── README.md
├── requirements.txt
└── LICENSE.txt

```

---

## Installation

### Requirements
- Python 3.9 or later  
- Recommended: create and activate a virtual environment

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows (PowerShell)
pip install --upgrade pip
pip install -r requirements.txt
```

**Minimum dependencies in `requirements.txt`:**

* flask
* numpy
* pandas
* dill

---

## Usage

### 1. (Optional) Train the model

To retrain the model, run the Jupyter notebook:

```bash
jupyter notebook notebook/train_model.ipynb
```

Executing the notebook generates `backend/model.pkl` (the trained model).

### 2. Run the application

From the repository root:

```bash
cd backend
python app.py
```

* Flask serves by default at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
* For production deployment, avoid using `debug=True`. Use a WSGI server (e.g., Gunicorn) and configure proper environment variables and security settings.

---

## API: `/predict`

**Endpoint** for programmatic access to predictions.

* **Request (JSON)**:

```json
{
    "age": 30,
    "height": 170,
    "weight": 70,
    "bp": 120,
    "glucose": 90,
    "activity": "normal",
    "diet": "healthy",
    "family_history": "yes"
}
```

* **Response (JSON)**:

```json
{
    "risk_level": 2
}
```

`risk_level` corresponds to a risk category:
`0 = Low, 1 = Potential, 2 = Medium, 3 = High, 4 = Danger`.

* **Error Codes**:

  * `400 Bad Request` — missing or invalid input fields
  * `500 Internal Server Error` — unexpected server-side error

---

## Data and Preprocessing

* Expected input features: `age`, `height (cm)`, `weight (kg)`, `bp`, `glucose`, `activity`, `diet`, `family_history`.
* Categorical inputs (`activity`, `diet`, `family_history`) are encoded numerically in `backend/utils.py`. Ensure frontend values match or use acceptable textual values.
* BMI is automatically calculated from height and weight.

---

## Evaluation & Limitations

* **Educational purposes only**; not a clinical tool.
* Simple Random Forest implementation with **no advanced production-grade features** (cross-validation, hyperparameter tuning, pruning, parallel training).
* Limited handling of missing values or categorical encodings — data should be validated before training or inference.
* Performance and scalability are lower compared to optimized libraries (e.g., scikit-learn).

---

## Recommended Improvements

* Use versioned model artifacts (e.g., JSON tree structures).
* Add automated unit tests for ML components and API endpoints.
* Implement robust input validation and logging.
* Add CI/CD pipelines and automated tests.
* Support configurable training via YAML/JSON and experiment logging.

---

## Contributing

You are welcome to contribute **suggestions, improvements, and bug reports**. 

> Note: Direct modifications, forks, or redistribution for commercial purposes are **not permitted**. Contributions should only aim to improve or fix the project.

To submit a suggestion or report a bug:

1. Open an Issue describing your suggestion or bug.
2. If relevant, propose a patch by referencing the Issue, but do **not** redistribute the project.
3. All contributions will be reviewed by the project owner before inclusion.

---

## License
This project is protected under a custom license. 

- Non-commercial use only.
- Contributions (suggestions, improvements, bug reports) are welcome.
- Redistribution or commercial use is prohibited.

See the [LICENSE.txt](LICENSE.txt) file for full details.

---

## Author

**Waseem Alyazidi**

---

## Disclaimer

This project is for **educational and demonstration purposes only**. It is **not medical advice** or a diagnostic tool. Always consult qualified health professionals for medical concerns.

```