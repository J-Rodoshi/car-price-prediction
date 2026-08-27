import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
import numpy as np

# ==========================
# Load saved files (with error handling)
# ==========================
try:
    model = joblib.load("xgb_car_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    model_available = True
except Exception as e:
    model = None
    scaler = None
    model_columns = None
    model_available = False
    print(f"Error loading model files: {e}")

# ==========================
# Original columns
# ==========================
cat_cols = ["name", "fuel", "seller_type", "transmission"]

num_cols = [
    "year", "km_driven", "owner", "mileage",
    "engine", "max_power", "seats"
]

# Scale ceiling in lakhs (₹). Values above this still render, the marker
# just pins at the right edge of the scale.
SCALE_MAX_LAKH = 30
ZONES = [
    ("Budget", 0, 5),
    ("Standard", 5, 15),
    ("Premium", 15, SCALE_MAX_LAKH),
]

# ==========================
# Small helpers for building form fields
# ==========================

def field(label, children, input_id=None):
    """Wrapper for a labelled form field."""
    label_component = html.Label(label, htmlFor=input_id) if input_id else html.Label(label)
    return html.Div([label_component, children], className="field")


def dropdown_field(label, id_, options, value):
    return field(
        label,
        dcc.Dropdown(
            id=id_,
            options=[{"label": o, "value": o} for o in options],
            value=value,
            clearable=False,
            className="dropdown"
        ),
        input_id=id_
    )


def number_field(label, id_, value, step=1, min_=None, max_=None):
    return field(
        label,
        dcc.Input(
            id=id_,
            type="number",
            value=value,
            step=step,
            min=min_,
            max=max_,
        ),
        input_id=id_
    )


def active_zone(price_lakh):
    for name, lo, hi in ZONES:
        if lo <= price_lakh < hi:
            return name
    return ZONES[-1][0]


# ==========================
# Validation helper
# ==========================
def validate_inputs(year, km_driven, mileage, engine, max_power, seats):
    """Return a list of error messages if inputs are out of range."""
    errors = []
    if not (1990 <= year <= 2025):
        errors.append("Year must be between 1990 and 2025.")
    if km_driven < 0 or km_driven > 1_000_000:
        errors.append("KM driven must be between 0 and 1,000,000.")
    if mileage <= 0 or mileage > 50:
        errors.append("Mileage must be between 0 and 50 kmpl.")
    if engine < 500 or engine > 4000:
        errors.append("Engine size must be between 500 and 4000 cc.")
    if max_power < 10 or max_power > 300:
        errors.append("Max power must be between 10 and 300 bhp.")
    if not (2 <= seats <= 9):
        errors.append("Seats must be between 2 and 9.")
    return errors


# ==========================
# Data preparation helper
# ==========================
def prepare_dataframe(name, year, km_driven, fuel, seller_type, transmission,
                      owner_label, mileage, engine, max_power, seats):
    """Build and preprocess the DataFrame for prediction."""
    owner = OWNER_MAP[owner_label]

    data = pd.DataFrame({
        "name": [name],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats],
    })

    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    data[num_cols] = scaler.transform(data[num_cols])
    data = data.reindex(columns=model_columns, fill_value=0)
    return data


# ==========================
# Dash App
# ==========================
app = dash.Dash(__name__, assets_folder='assets')
app.title = "Car Price Prediction"

BRANDS = ["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "BMW", "Audi"]

# Layout
app.layout = html.Div(className="page", children=[

    html.P("Automated Valuation", className="eyebrow"),
    html.H1("What is your car worth today?", className="headline"),
    html.P(
        "Enter the vehicle's specifications and listing details. "
        "The estimate is produced by a trained regression model and "
        "presented as a single reference value.",
        className="subhead"
    ),

    html.Div(className="grid", children=[

        # ---------------- Left: assessment form ----------------
        html.Div(className="panel", children=[
            html.P("Vehicle Assessment", className="panel-title"),
            html.P("All fields are required for a prediction.", className="panel-sub"),

            html.P("Identity", className="section-label"),
            html.Div(className="field-group", children=[
                dropdown_field("Brand", "name", BRANDS, "Maruti"),
                number_field("Year", "year", 2015, step=1, min_=1990, max_=2025),
            ]),
            html.Div(className="field-group", children=[
                number_field("KM driven", "km_driven", 50000, step=500, min_=0, max_=1000000),
                dropdown_field("Owner", "owner_label", ["First", "Second", "Third", "Fourth+"], "First"),
            ]),

            html.Hr(className="section-divider"),
            html.P("Powertrain", className="section-label"),
            html.Div(className="field-group", children=[
                dropdown_field("Fuel", "fuel", ["Diesel", "Petrol"], "Diesel"),
                dropdown_field("Transmission", "transmission", ["Manual", "Automatic"], "Manual"),
            ]),
            html.Div(className="field-group", children=[
                number_field("Engine (cc)", "engine", 1200, step=50, min_=500, max_=4000),
                number_field("Max power (bhp)", "max_power", 80, step=1, min_=10, max_=300),
            ]),
            html.Div(className="field-group", children=[
                number_field("Mileage (kmpl)", "mileage", 20, step=0.1, min_=0, max_=50),  # <-- FIXED
                number_field("Seats", "seats", 5, step=1, min_=2, max_=9),
            ]),

            html.Hr(className="section-divider"),
            html.P("Listing", className="section-label"),
            html.Div(className="field-group full", children=[
                dropdown_field(
                    "Seller type", "seller_type",
                    ["Individual", "Dealer", "Trustmark Dealer"], "Individual"
                ),
            ]),

            dcc.Loading(
                id="loading-predict",
                type="circle",
                children=html.Button("Generate estimate", id="predict_button", className="predict-btn", n_clicks=0)
            ),
        ]),

        # ---------------- Right: valuation certificate ----------------
        html.Div(className="panel certificate", children=[
            html.P("Estimated Market Value", className="cert-eyebrow"),

            html.Div(id="prediction", className="cert-value", role="status", **{"aria-live": "polite"}, children=[
                html.Span("Awaiting details", className="placeholder")
            ]),

            html.Hr(className="cert-rule"),

            html.Div(className="scale", children=[
                html.Div(className="scale-track", children=[
                    html.Div(id="scale_fill", className="scale-fill", style={"width": "0%"}),
                    html.Div(id="scale_marker", className="scale-marker", style={"left": "0%"}),
                ]),
                html.Div(id="scale_labels", className="scale-labels", children=[
                    html.Span("Budget"),
                    html.Span("Standard"),
                    html.Span("Premium"),
                ]),
            ]),

            html.Span(id="status_flag", className="status-pill", children=[
                html.Span(className="dot"), "Waiting for input"
            ]),
        ]),
    ]),
])

# ==========================
# Prediction callback
# ==========================
OWNER_MAP = {"First": 1, "Second": 2, "Third": 3, "Fourth+": 4}


@app.callback(
    Output("prediction", "children"),
    Output("scale_fill", "style"),
    Output("scale_marker", "style"),
    Output("scale_labels", "children"),
    Output("status_flag", "children"),
    Output("status_flag", "className"),
    Input("predict_button", "n_clicks"),
    State("name", "value"),
    State("year", "value"),
    State("km_driven", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("owner_label", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
)
def predict(n, name, year, km_driven, fuel, seller_type, transmission,
            owner_label, mileage, engine, max_power, seats):

    zone_labels = [html.Span(zname) for zname, _, _ in ZONES]

    if not model_available:
        return (
            [html.Span("Model unavailable", className="placeholder")],
            {"width": "0%"},
            {"left": "0%"},
            zone_labels,
            [html.Span(className="dot"), "Model files missing"],
            "status-pill error"
        )

    if not n:
        return (
            [html.Span("Awaiting details", className="placeholder")],
            {"width": "0%"},
            {"left": "0%"},
            zone_labels,
            [html.Span(className="dot"), "Waiting for input"],
            "status-pill"
        )

    # Check for missing values
    required = [name, year, km_driven, fuel, seller_type, transmission,
                owner_label, mileage, engine, max_power, seats]
    if any(v is None for v in required):
        return (
            [html.Span("Fill every field", className="placeholder")],
            {"width": "0%"},
            {"left": "0%"},
            zone_labels,
            [html.Span(className="dot"), "Missing fields"],
            "status-pill error"
        )

    # Validate ranges
    errors = validate_inputs(year, km_driven, mileage, engine, max_power, seats)
    if errors:
        error_msg = " • ".join(errors) if len(errors) <= 2 else f"{errors[0]} (and {len(errors)-1} more)"
        return (
            [html.Span("Check your inputs", className="placeholder")],
            {"width": "0%"},
            {"left": "0%"},
            zone_labels,
            [html.Span(className="dot"), error_msg],
            "status-pill error"
        )

    # Prepare data and predict
    data = prepare_dataframe(name, year, km_driven, fuel, seller_type, transmission,
                             owner_label, mileage, engine, max_power, seats)

    log_prediction = model.predict(data)[0]
    actual_price = np.expm1(log_prediction)
    price_lakh = actual_price / 100000

    position_pct = min(price_lakh, SCALE_MAX_LAKH) / SCALE_MAX_LAKH * 100
    current_zone = active_zone(price_lakh)

    zone_labels = [
        html.Span(zname, className="active" if zname == current_zone else "")
        for zname, _, _ in ZONES
    ]

    readout = [f"\u20b9 {actual_price:,.0f}"]
    status = [html.Span(className="dot"), "Estimate ready"]

    return (
        readout,
        {"width": f"{position_pct}%"},
        {"left": f"{position_pct}%"},
        zone_labels,
        status,
        "status-pill ready"
    )


if __name__ == "__main__":
    app.run(debug=True)