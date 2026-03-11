import pathlib

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
COLUMNS_PATH = MODEL_DIR / "model_columns.joblib"


@st.cache_resource
def load_artifacts():
    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and COLUMNS_PATH.exists()):
        missing = [
            p.name
            for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]
            if not p.exists()
        ]
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing)
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    return model, scaler, model_columns


def predict_churn(customer_data, model, scaler, model_columns):
    input_df = pd.DataFrame([customer_data])
    categorical_cols_for_pred = ["Geography", "Gender"]
    input_df = pd.get_dummies(input_df, columns=categorical_cols_for_pred, drop_first=True)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    num_cols_to_scale = [
        "CredRate",
        "Age",
        "Tenure",
        "Balance",
        "Prod Number",
        "HasCrCard",
        "ActMem",
        "EstimatedSalary",
    ]
    input_df[num_cols_to_scale] = scaler.transform(input_df[num_cols_to_scale])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return ("CHURN (Yes)" if prediction == 1 else "STAY (No)"), probability


def reset_for_new_prediction():
    st.session_state.show_prediction = False
    st.session_state.prediction = None
    st.session_state.pending_step = 1


st.set_page_config(page_title="Credit Churn Predictor", layout="wide")

LIGHT_VARS = {
    "color-scheme": "light",
    "--bg": "#f8fafc",
    "--panel": "#ffffff",
    "--card": "#f1f5f9",
    "--accent": "#f59e0b",
    "--accent-2": "#0284c7",
    "--text": "#0f172a",
    "--muted": "#64748b",
    "--ok": "#16a34a",
    "--hero": "linear-gradient(120deg, #ffffff 0%, #e2e8f0 60%)",
    "--app-bg": "radial-gradient(1200px 600px at 10% 0%, #f8fafc, #e2e8f0 60%)",
    "--field-bg": "#ffffff",
    "--field-border": "#cbd5e1",
}

DARK_VARS = {
    "color-scheme": "dark",
    "--bg": "#0f1115",
    "--panel": "#151a21",
    "--card": "#1b2230",
    "--accent": "#f59e0b",
    "--accent-2": "#38bdf8",
    "--text": "#e5e7eb",
    "--muted": "#9aa4b2",
    "--ok": "#22c55e",
    "--hero": "linear-gradient(120deg, #1b2230 0%, #111827 60%)",
    "--app-bg": "radial-gradient(1200px 600px at 10% 0%, #1b2230, #0f1115 60%)",
    "--field-bg": "#111827",
    "--field-border": "#273244",
}


def _vars_to_css(vars_dict):
    return "\n".join([f"{k}: {v};" for k, v in vars_dict.items()])

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

if st.session_state.theme_mode == "Light":
    root_css = f":root {{\n{_vars_to_css(LIGHT_VARS)}\n}}"
else:
    root_css = f":root {{\n{_vars_to_css(DARK_VARS)}\n}}"

st.markdown(
    f"""
    <style>
    {root_css}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {{
        background: var(--app-bg) !important;
        color: var(--text) !important;
    }}
    html, body {{
        background: var(--app-bg) !important;
        color: var(--text) !important;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background: var(--panel) !important;
        border-right: 1px solid var(--field-border) !important;
    }}
    [data-testid="stSidebar"] *, [data-testid="stSidebarContent"] *,
    [data-testid="stAppViewContainer"] * {{
        color: var(--text);
    }}
    label, p, span, div, h1, h2, h3, h4, h5, h6 {{
        color: var(--text) !important;
    }}
    .hero {{
        background: var(--hero);
        border: 1px solid var(--field-border);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }}
    .hero h1 {{ font-size: 2.2rem; margin-bottom: 6px; }}
    .hero p {{ color: var(--muted) !important; margin: 0; }}
    .step-card {{
        background: var(--card) !important;
        border: 1px solid var(--field-border) !important;
        border-radius: 16px;
        padding: 18px;
    }}
    .badge {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        background: rgba(56, 189, 248, 0.15);
        color: var(--accent-2) !important;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }}
    .metric-card {{
        background: var(--panel) !important;
        border: 1px solid var(--field-border) !important;
        border-radius: 14px;
        padding: 14px 16px;
    }}
    .result-card {{
        background: var(--panel) !important;
        border: 1px solid var(--field-border) !important;
        border-radius: 16px;
        padding: 18px 20px;
    }}
    .result-title {{
        font-size: 1.6rem;
        margin: 0;
    }}
    .result-sub {{
        color: var(--muted) !important;
        margin: 4px 0 10px 0;
    }}
    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.8rem;
        border: 1px solid var(--field-border);
        color: var(--muted) !important;
        background: var(--card) !important;
    }}
    .score-grid {{
        display: grid;
        grid-template-columns: 1fr 140px;
        gap: 16px;
        align-items: center;
        margin-top: 12px;
    }}
    .score-box {{
        border: 1px solid var(--field-border);
        border-radius: 12px;
        padding: 10px 12px;
        text-align: center;
        background: var(--card) !important;
    }}
    .score-box strong {{
        font-size: 1.2rem;
    }}
    .cta {{
        background: linear-gradient(90deg, #f59e0b, #f97316);
        color: #0b0b0b !important;
        border-radius: 12px;
        padding: 10px 16px;
        font-weight: 600;
    }}
    .stButton>button, 
    .stFormSubmitButton>button, 
    [data-testid="baseButton-secondary"], 
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondaryFormSubmit"],
    button[kind="secondaryFormSubmit"],
    button[kind="primaryFormSubmit"],
    button[kind="primary"],
    button {{
        background: linear-gradient(90deg, #f59e0b, #f97316) !important;
        background-image: linear-gradient(90deg, #f59e0b, #f97316) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 0 !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
    }}
    .modal-actions {{
        display: flex;
        gap: 12px;
        margin-top: 16px;
        justify-content: flex-end;
    }}
    .modal-actions .stButton>button {{
        min-width: 140px;
    }}
    .stTextInput>div>div>input,
    .stNumberInput input,
    .stSelectbox>div>div>div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input[type="number"],
    input[type="text"] {{
        background-color: var(--field-bg) !important;
        border: 1px solid var(--field-border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
    }}
    div[data-baseweb="input"] button,
    div[data-testid="stNumberInputContainer"] button,
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] {{
        background-color: transparent !important;
        color: var(--text) !important;
    }}
    div[data-baseweb="input"] input {{
        color: var(--text) !important;
        background-color: transparent !important;
    }}
    div[data-baseweb="select"] span {{
        color: var(--text) !important;
    }}
    /* Dropdown menu background fix */
    div[role="listbox"], ul[role="listbox"] {{
        background-color: var(--panel) !important;
    }}
    li[role="option"] {{
        background-color: transparent !important;
        color: var(--text) !important;
    }}
    .sidebar-title {{ font-weight: 700; letter-spacing: 0.5px; }}
    .progress {{
        height: 8px;
        background-color: var(--field-bg) !important;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid var(--field-border);
    }}
    .progress > div {{
        height: 100%;
        background: linear-gradient(90deg, #38bdf8, #22c55e);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="badge">Guided Flow</div>
      <h1>Credit Churn Predictor</h1>
      <p>Step-by-step input, instant prediction, and clear risk signal.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    loaded_model, loaded_scaler, model_columns = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Train and save artifacts to the model folder, for example:\n"
        "python train_model.py --csv data\\Churn_Modelling.csv"
    )
    st.stop()

if "step" not in st.session_state:
    st.session_state.step = 1
if "customer" not in st.session_state:
    st.session_state.customer = {}
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None

steps = ["Profile", "Financials", "Account", "Review"]

if "pending_step" in st.session_state:
    st.session_state.step = st.session_state.pending_step
    st.session_state.nav_step_widget = steps[st.session_state.step - 1]
    del st.session_state.pending_step

if "nav_step_widget" not in st.session_state:
    st.session_state.nav_step_widget = steps[st.session_state.step - 1]

with st.sidebar:
    if st.session_state.theme_mode not in ("Dark", "Light"):
        st.session_state.theme_mode = "Dark"
    theme_mode = st.selectbox(
        "Theme",
        ["Dark", "Light"],
        index=["Dark", "Light"].index(st.session_state.theme_mode),
    )
    st.session_state.theme_mode = theme_mode
    st.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)
    step_choice = st.radio("Go to step", steps, key="nav_step_widget")
    if step_choice != steps[st.session_state.step - 1]:
        st.session_state.step = steps.index(step_choice) + 1
    progress = int((st.session_state.step / len(steps)) * 100)
    st.markdown("Progress")
    st.markdown(
        f"<div class='progress'><div style='width:{progress}%;'></div></div>",
        unsafe_allow_html=True,
    )
    st.caption("Complete each step to unlock prediction.")

col_left, col_right = st.columns([2.1, 1.2], gap="large")

with col_right:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Quick Tips")
    st.write("Use realistic values from your CRM or onboarding form.")
    st.write("Higher churn probability indicates higher risk.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_left:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    if st.session_state.step == 1:
        st.subheader("Step 1 - Profile")
        with st.form("step1"):
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            submitted = st.form_submit_button("Next")
        if submitted:
            st.session_state.customer.update(
                {"Geography": geography, "Gender": gender, "Age": age}
            )
            st.session_state.pending_step = 2
            st.rerun()

    elif st.session_state.step == 2:
        st.subheader("Step 2 - Financials")
        with st.form("step2"):
            cred_rate = st.number_input(
                "Credit Rating", min_value=0, max_value=1000, value=600, step=1
            )
            balance = st.number_input(
                "Balance", min_value=0.0, value=100000.0, step=1000.0, format="%.2f"
            )
            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                format="%.2f",
            )
            cols = st.columns(2)
            with cols[0]:
                back = st.form_submit_button("Back")
            with cols[1]:
                next_step = st.form_submit_button("Next")
        if back:
            st.session_state.pending_step = 1
            st.rerun()
        if next_step:
            st.session_state.customer.update(
                {
                    "CredRate": cred_rate,
                    "Balance": balance,
                    "EstimatedSalary": estimated_salary,
                }
            )
            st.session_state.pending_step = 3
            st.rerun()

    elif st.session_state.step == 3:
        st.subheader("Step 3 - Account")
        with st.form("step3"):
            tenure = st.number_input(
                "Tenure (years)", min_value=0, max_value=20, value=5, step=1
            )
            prod_number = st.number_input(
                "Product Number", min_value=1, max_value=4, value=1, step=1
            )
            st.caption("For the fields below: 1 = Yes, 0 = No")
            has_cr_card = st.selectbox("Has Credit Card", [0, 1], index=1)
            act_mem = st.selectbox("Active Member", [0, 1], index=1)
            cols = st.columns(2)
            with cols[0]:
                back = st.form_submit_button("Back")
            with cols[1]:
                next_step = st.form_submit_button("Next")
        if back:
            st.session_state.pending_step = 2
            st.rerun()
        if next_step:
            st.session_state.customer.update(
                {
                    "Tenure": tenure,
                    "Prod Number": prod_number,
                    "HasCrCard": has_cr_card,
                    "ActMem": act_mem,
                }
            )
            st.session_state.pending_step = 4
            st.rerun()

    else:
        st.subheader("Step 4 - Review & Predict")
        customer = st.session_state.customer
        required_keys = [
            "Geography",
            "Gender",
            "Age",
            "CredRate",
            "Balance",
            "EstimatedSalary",
            "Tenure",
            "Prod Number",
            "HasCrCard",
            "ActMem",
        ]
        missing = [k for k in required_keys if k not in customer]
        if missing:
            st.warning("Missing fields: " + ", ".join(missing))
        st.write(customer)
        with st.form("step4"):
            cols = st.columns(2)
            with cols[0]:
                back = st.form_submit_button("Back")
            with cols[1]:
                predict = st.form_submit_button("Predict", disabled=bool(missing))
        if back:
            st.session_state.pending_step = 3
            st.rerun()
        if predict and not missing:
            result, prob = predict_churn(
                customer, loaded_model, loaded_scaler, model_columns
            )
            st.session_state.prediction = (result, prob)
            st.session_state.show_prediction = True

def render_prediction_content(result, prob):
    risk_label = "High Risk" if prob >= 0.6 else "Moderate Risk" if prob >= 0.35 else "Low Risk"
    accent = "#ef4444" if prob >= 0.6 else "#f59e0b" if prob >= 0.35 else "#22c55e"
    st.markdown(
        f"""
        <div class="result-card" style="border-color:{accent};">
          <div class="pill" style="border-color:{accent}; color:{accent};">Prediction</div>
          <h2 class="result-title">{result}</h2>
          <div class="result-sub">Risk tier: <strong style="color:{accent};">{risk_label}</strong></div>
          <div class="score-grid">
            <div>
              <div class="progress"><div style="width:{int(prob*100)}%; background:{accent};"></div></div>
              <div class="result-sub" style="margin-top:8px;">Probability of Churn</div>
            </div>
            <div class="score-box">
              <div class="result-sub">Score</div>
              <strong>{prob:.2%}</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if st.session_state.show_prediction and st.session_state.prediction:
    result, prob = st.session_state.prediction

    if hasattr(st, "dialog"):
        @st.dialog("Prediction Result")
        def prediction_dialog():
            render_prediction_content(result, prob)
            st.markdown('<div class="modal-actions">', unsafe_allow_html=True)
            if st.button("Predict Another"):
                reset_for_new_prediction()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        prediction_dialog()

    elif hasattr(st, "experimental_dialog"):
        @st.experimental_dialog("Prediction Result")
        def prediction_dialog():
            render_prediction_content(result, prob)
            st.markdown('<div class="modal-actions">', unsafe_allow_html=True)
            if st.button("Predict Another"):
                reset_for_new_prediction()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        prediction_dialog()

    else:
        st.subheader("Prediction Result")
        render_prediction_content(result, prob)
        st.markdown('<div class="modal-actions">', unsafe_allow_html=True)
        if st.button("Predict Another"):
            reset_for_new_prediction()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
