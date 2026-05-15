import json
from pathlib import Path

import pandas as pd


class DataLeakageAudit:
    def run(self, df, features, target_column: str = "target") -> dict:
        warnings = []
        critical = []
        feature_names = list(features)
        for feature in feature_names:
            lower = feature.lower()
            if "future" in lower:
                critical.append(f"Feature appears to include future information: {feature}")
            if lower == target_column.lower():
                critical.append("Target column is included in features.")
            if "shift(-" in lower or "lead" in lower:
                critical.append(f"Feature name suggests negative shift: {feature}")
        if "timestamp" in df.columns and not pd.to_datetime(df["timestamp"]).is_monotonic_increasing:
            warnings.append("Timestamps are not sorted.")
        if df.duplicated().any():
            warnings.append("Duplicated rows found.")
        if target_column in df.columns:
            for feature in feature_names:
                if feature in df.columns:
                    corr = pd.to_numeric(df[feature], errors="coerce").corr(pd.to_numeric(df[target_column], errors="coerce"))
                    if pd.notna(corr) and abs(corr) > 0.98:
                        warnings.append(f"Feature has suspiciously high correlation with target: {feature}")
        result = {"has_critical_leakage": bool(critical), "critical": critical, "warnings": warnings, "status": "ERROR" if critical else "OK"}
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/logs/data_leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
