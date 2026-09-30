from pathlib import Path
import os
import joblib
import pandas as pd
import requests
import streamlit as st

import production_pipeline  # Available if the saved model uses this module.


PROJECT_FOLDER = Path(__file__).resolve().parent
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MODEL_FILE = PROJECT_FOLDER / "final_production_candidate.joblib"

DATASET_OPTIONS = [
    PROJECT_FOLDER / "final_modeling_dataset.csv",
    PROJECT_FOLDER / "final_modeling_dataset(1).csv",
    PROJECT_FOLDER / "final_feature_set.csv",
]

NUMERICAL_FEATURES = [
    "production_rate_per_hour",
    "good_units",
    "units_produced",
    "shift_start_hour",
    "estimated_oee_pct",
    "downtime_minutes",
    "runtime_hours",
    "defect_rate_pct",
    "experience_runtime_interaction",
    "temperature_deviation",
    "defect_count",
    "experience_level",
    "cycle_time_avg",
    "machine_prior_avg_downtime",
    "machine_prior_avg_defect_rate",
    "machine_prior_avg_units",
    "maintenance_downtime",
    "humidity",
]

CATEGORICAL_FEATURES = ["shift_name", "machine_id"]
TARGET = "shift_efficiency_score"


@st.cache_data
def load_data():
    dataset_file = next(
        (path for path in DATASET_OPTIONS if path.is_file()),
        None,
    )

    if dataset_file is None:
        raise FileNotFoundError(
            "No final modeling dataset was found beside the dashboard."
        )

    data = pd.read_csv(dataset_file).drop_duplicates()
    return data, dataset_file.name


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def api_is_ready():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("model_loaded") is True
    except (requests.RequestException, ValueError):
        return False


st.set_page_config(
    page_title="NorDex Manufacturing Intelligence",
    page_icon="🏭",
    layout="wide",
)

st.title("NorDex Manufacturing Intelligence")
st.caption("Week 3 Day 3 | Operational analysis and shift efficiency prediction")

try:
    data, dataset_name = load_data()
except (FileNotFoundError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

needed = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET])
missing = sorted(needed - set(data.columns))

if missing:
    st.error(f"The dashboard dataset is missing: {missing}")
    st.stop()

st.sidebar.header("Filters")

shift_options = sorted(data["shift_name"].dropna().astype(str).unique())
machine_options = sorted(data["machine_id"].dropna().astype(str).unique())

selected_shifts = st.sidebar.multiselect(
    "Shift",
    options=shift_options,
    default=shift_options,
)

selected_machines = st.sidebar.multiselect(
    "Machine",
    options=machine_options,
    default=machine_options,
)

filtered = data[
    data["shift_name"].astype(str).isin(selected_shifts)
    & data["machine_id"].astype(str).isin(selected_machines)
].copy()

st.sidebar.caption(f"Historical dataset: {dataset_name}")

if api_is_ready():
    st.sidebar.success("Prediction API is ready")
else:
    st.sidebar.warning(
        "Prediction API is offline. Start the Day 2 API to make predictions."
    )

overview_tab, insights_tab, prediction_tab = st.tabs(
    ["Performance overview", "Model insights", "New prediction"]
)

with overview_tab:
    st.subheader("Manufacturing performance")

    if filtered.empty:
        st.info("No records match the current filters.")
    else:
        metric_columns = st.columns(4)

        metric_columns[0].metric(
            "Historical records",
            f"{len(filtered):,}",
        )
        metric_columns[1].metric(
            "Average efficiency score",
            f"{filtered[TARGET].mean():.2f}",
        )
        metric_columns[2].metric(
            "Average estimated OEE",
            f"{filtered['estimated_oee_pct'].mean():.2f}%",
        )
        metric_columns[3].metric(
            "Average downtime",
            f"{filtered['downtime_minutes'].mean():.2f} min",
        )

        left, right = st.columns(2)

        with left:
            st.markdown("#### Average efficiency by shift")
            by_shift = (
                filtered.groupby("shift_name")[TARGET]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(by_shift)

            st.markdown("#### Average downtime by shift")
            by_downtime = (
                filtered.groupby("shift_name")["downtime_minutes"]
                .mean()
            )
            st.bar_chart(by_downtime)

        with right:
            st.markdown("#### Average efficiency by machine")
            by_machine = (
                filtered.groupby("machine_id")[TARGET]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(by_machine)

            st.markdown("#### Average defect rate by shift")
            by_defect = (
                filtered.groupby("shift_name")["defect_rate_pct"]
                .mean()
            )
            st.bar_chart(by_defect)

        st.markdown("#### Filtered manufacturing records")
        st.dataframe(
            filtered.head(100),
            use_container_width=True,
        )

        st.download_button(
            "Download filtered records",
            data=filtered.to_csv(index=False),
            file_name="week3_day3_filtered_manufacturing_records.csv",
            mime="text/csv",
        )

with insights_tab:
    st.subheader("Model insights")

    try:
        saved_model = load_model()
        preprocessor = saved_model.named_steps["preprocessor"]
        estimator = saved_model.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()
        importances = estimator.feature_importances_

        importance_table = (
            pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances,
            })
            .sort_values("Importance", ascending=False)
            .head(10)
        )

        st.markdown("#### Ten most important model features")
        st.bar_chart(
            importance_table.set_index("Feature")["Importance"]
        )
        st.dataframe(
            importance_table,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Feature importance describes how the fitted model uses "
            "predictors. It does not establish that changing a feature "
            "will cause efficiency to change."
        )
    except (FileNotFoundError, KeyError, AttributeError, ValueError) as exc:
        st.error(f"Could not load model insights: {exc}")

    if not filtered.empty:
        st.markdown("#### Operational observations")

        highest_downtime_shift = (
            filtered.groupby("shift_name")["downtime_minutes"]
            .mean()
            .idxmax()
        )
        highest_defect_shift = (
            filtered.groupby("shift_name")["defect_rate_pct"]
            .mean()
            .idxmax()
        )

        st.write(
            f"- Review downtime patterns for the **{highest_downtime_shift}** "
            "shift, which has the highest average downtime among the "
            "currently filtered records."
        )
        st.write(
            f"- Review quality checks for the **{highest_defect_shift}** "
            "shift, which has the highest average defect rate among the "
            "currently filtered records."
        )
        st.caption(
            "These are prompts for investigation based on historical "
            "averages, not evidence that a shift caused the outcome."
        )

with prediction_tab:
    st.subheader("Predict shift efficiency")
    st.write(
        "Start with values from a historical record, adjust them, "
        "then submit the record to the Day 2 API."
    )

    reference = data[
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    ].dropna()

    if reference.empty:
        st.error(
            "No complete reference row is available for the input form."
        )
        st.stop()

    initial = reference.iloc[0]

    with st.form("prediction_form"):
        shift_value = st.selectbox(
            "Shift name",
            options=shift_options,
            index=shift_options.index(str(initial["shift_name"])),
        )

        machine_value = st.selectbox(
            "Machine ID",
            options=machine_options,
            index=machine_options.index(str(initial["machine_id"])),
        )

        st.markdown("#### Numerical manufacturing inputs")
        input_columns = st.columns(3)
        numeric_values = {}

        for index, feature in enumerate(NUMERICAL_FEATURES):
            with input_columns[index % 3]:
                numeric_values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(initial[feature]),
                    key=f"input_{feature}",
                )

        submitted = st.form_submit_button("Predict efficiency")

    if submitted:
        record = {
            **numeric_values,
            "shift_name": shift_value,
            "machine_id": machine_value,
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"record": record},
                timeout=15,
            )

            if response.status_code == 200:
                score = response.json()[
                    "predicted_shift_efficiency_score"
                ]
                st.success(f"Predicted Shift Efficiency Score: {score:.2f}")

                if score < 70:
                    st.info(
                        "Review downtime, production rate, and quality "
                        "indicators for this record."
                    )
                else:
                    st.info(
                        "Compare this prediction with actual shift "
                        "performance when the outcome becomes available."
                    )
            else:
                st.error(
                    f"API returned HTTP {response.status_code}: "
                    f"{response.text}"
                )

        except requests.RequestException as exc:
            st.error(
                "Could not reach the Day 2 prediction API. "
                f"Connection details: {exc}"
            )
