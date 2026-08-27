# 🚗 Car Price Prediction System

This project predicts the selling price of a used car based on its features like brand, year, kilometers driven, fuel type, transmission, engine size, and more.  
The system is built with **machine learning** and includes a **web application** where users can enter a car's details and get an instant price estimate.

The project includes:

- A Jupyter Notebook containing the complete machine learning workflow
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Multiple regression models
- XGBoost model selection
- Saved model and preprocessing files
- A Dash web application for making predictions
- Docker support for running the web application easily

---

## 📖 What This Project Does

Imagine you want to sell your car but don't know how much to ask for.  
This tool helps by taking information about the car and giving an estimated market value.

The prediction is made by a **computer model** that has "learned" from thousands of real used-car listings.  
The model looks at patterns in the data — like how age, mileage, engine power, and brand affect price — and uses those patterns to estimate the price of a new car.

---


## 🧠 How It Works

The project follows these steps:

1. Load the used-car dataset.
2. Inspect and clean the data.
3. Perform exploratory data analysis.
4. Select and encode features.
5. Split the data into training and test sets.
6. Handle missing values.
7. Scale the features.
8. Train several machine learning models.
9. Compare model performance.
10. Select XGBoost as the best model.
11. Save the model and preprocessing objects.
12. Build a Dash web application for real-time predictions.

---

##  📊The Dataset

The dataset (`Cars.csv`) contains the following information for each car:

- `name` – the manufacturer (e.g., Maruti, Hyundai, BMW)
- `year` – year of manufacture
- `selling_price` – the price the car was sold for (target)
- `km_driven` – total distance driven (in kilometres)
- `fuel` – Diesel, Petrol, etc.
- `seller_type` – Individual, Dealer, Trustmark Dealer
- `transmission` – Manual or Automatic
- `owner` – number of previous owners
- `mileage` – fuel efficiency (km per litre)
- `engine` – engine size (in CC)
- `max_power` – engine power (in bhp)
- `seats` – number of seats

Some data cleaning steps were applied:

- Removed CNG/LPG cars (they use different mileage units).
- Removed test drive cars (their prices are unrealistically high).
- Converted text values (like "1248 CC") into numbers.
- Log-transformed the price to make predictions more stable.

---

## 🧪 Model Performance

Several models were compared. The results (on the test set) are:

| Model               | R² Score | RMSE (log scale) | Meaning                          |
|---------------------|----------|------------------|----------------------------------|
| **XGBoost**         | **0.945**| **0.198**        | Best performer                   |
| Random Forest       | 0.943    | 0.202            | Very close second                |
| Gradient Boosting   | 0.934    | 0.218            | Good, but slightly less accurate |
| Ridge Regression    | 0.910    | 0.254            | Linear model                     |
| Lasso Regression    | ~0       | 0.844            | Performed poorly                 |

**XGBoost** was selected as the final model. It can predict prices with an average error of about **15%** – good enough for a quick estimate.

---


## 💻 How to Run the Web Application

### Option 1: Run Locally (without Docker)

**Requirements:** Python 3.8+ installed on your computer.

1. **Download or clone this repository**  
   ```bash
   git clone https://github.com/J-Rodoshi/car-price-prediction.git
   cd car-price-prediction
   ```

2. **Install required packages**  
   Open a terminal in the project folder and run:
   ```bash
   pip install -r app/requirements.txt
   ```

3. **Run the app**  
   ```bash
   python app/code/app.py
   ```

4. **Open your browser** and go to:  
   `http://127.0.0.1:8050`

You’ll see a form where you can enter a car’s details and click **"Generate estimate"**.

---

### Option 2: Run with Docker

If you have Docker installed, the app can be started without installing Python packages manually.

1. **Navigate to the project root** (where `docker-compose.yaml` is located).

2. **Start the container**:
   ```bash
   docker compose up --build
   ```

3. **Open your browser** and go to `http://localhost:8050`.

---


## 📁 Repository Structure
The repository is organised as follows:


car-price-prediction/
│
├── A1.ipynb                  # Jupyter notebook (full ML workflow)
├── A1.pdf                    # PDF export of the notebook
├── Cars.csv                  # Dataset
├── README.md                 # This file
├── docker-compose.yaml       # Docker Compose configuration
│
└── app/
    ├── Dockerfile            # Docker image definition
    ├── requirements.txt      # Python dependencies
    └── code/                 # Application source code
        ├── app.py            # Dash web app
        ├── assets/
        │   └── style.css     # Custom styling
        ├── xgb_car_price_model.pkl   # Trained XGBoost model
        ├── scaler.pkl        # Feature scaler
        └── model_columns.pkl # Column order after encoding

---


## 🛠️ Technologies Used

The project was developed using:

Python — programming language
Pandas — data manipulation
NumPy — numerical operations
Matplotlib — data visualization
Seaborn — statistical visualization
Scikit-learn — preprocessing and machine learning
XGBoost — final regression model
Joblib — model serialization
Dash — web application
Docker — application containerization
Jupyter Notebook — analysis and experimentation

---


## 🧑‍💻 For Developers – Reproducing the Notebook

1. Open `A1.ipynb` in Jupyter Notebook or JupyterLab.
2. Run all cells in order.
3. The notebook will:
   - Load and clean the data.
   - Train and compare several models.
   - Save the best model and preprocessing objects.
   - Demonstrate a sample prediction.

**Note:** The notebook is fully documented with explanations in each section.

---


## 📄 License

This project is created for educational purposes as part of a machine learning assignment.  
Feel free to use and modify it.

---


## 🙏 Acknowledgements

- Dataset: Car details dataset provided in the course.
- Tools: Python, pandas, scikit-learn, XGBoost, Dash, Docker.
