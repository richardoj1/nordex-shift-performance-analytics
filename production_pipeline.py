
from datetime import datetime, timezone

import numpy as np
import pandas as pd


class ManufacturingPredictionPipeline:
    """Validate manufacturing records and generate efficiency predictions."""

    def __init__(
        self,
        model,
        expected_columns,
        numerical_columns,
        categorical_columns
    ):
        self.model = model
        self.expected_columns = list(expected_columns)
        self.numerical_columns = list(numerical_columns)
        self.categorical_columns = list(categorical_columns)

    def validate(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame or dictionary."
            )

        if data.empty:
            raise ValueError("Input data contains no rows.")

        missing_columns = [
            column
            for column in self.expected_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        validated_data = data[
            self.expected_columns
        ].copy()

        for column in self.numerical_columns:
            original_non_null = validated_data[column].notna()

            converted = pd.to_numeric(
                validated_data[column],
                errors="coerce"
            )

            invalid_values = (
                original_non_null
                & converted.isna()
            ).sum()

            if invalid_values > 0:
                raise ValueError(
                    f"Column '{column}' contains "
                    f"{invalid_values} non-numeric value(s)."
                )

            validated_data[column] = converted

        for column in self.categorical_columns:
            validated_data[column] = (
                validated_data[column]
                .astype("string")
            )

        non_negative_columns = [
            "units_produced",
            "good_units",
            "defect_count",
            "runtime_hours",
            "downtime_minutes",
            "cycle_time_avg",
            "production_rate_per_hour"
        ]

        for column in non_negative_columns:
            if (
                column in validated_data.columns
                and (validated_data[column].dropna() < 0).any()
            ):
                raise ValueError(
                    f"Column '{column}' cannot contain "
                    "negative values."
                )

        if {
            "good_units",
            "units_produced"
        }.issubset(validated_data.columns):
            invalid_good_units = (
                validated_data["good_units"]
                > validated_data["units_produced"]
            )

            if invalid_good_units.any():
                raise ValueError(
                    "good_units cannot exceed units_produced."
                )

        if "humidity" in validated_data.columns:
            invalid_humidity = ~validated_data[
                "humidity"
            ].dropna().between(0, 100)

            if invalid_humidity.any():
                raise ValueError(
                    "humidity must be between 0 and 100."
                )

        return validated_data

    def predict(self, data):
        validated_data = self.validate(data)

        predictions = self.model.predict(
            validated_data
        )

        results = pd.DataFrame({
            "predicted_shift_efficiency_score":
                np.asarray(predictions, dtype=float)
        })

        results["prediction_timestamp_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )

        results["pipeline_status"] = "Successful"

        return results
