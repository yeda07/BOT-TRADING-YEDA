import pandas as pd

from app.validation.data_leakage_audit import DataLeakageAudit


def test_data_leakage_audit_detects_future_feature():
    result = DataLeakageAudit().run(pd.DataFrame({"future_close": [1, 2], "target": [0, 1]}), ["future_close"])

    assert result["has_critical_leakage"]
    assert "future" in result["critical"][0]
