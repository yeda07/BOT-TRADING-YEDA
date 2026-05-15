from sklearn.ensemble import RandomForestClassifier

from app.market.features import create_features
from app.validation.stress_testing import StressTester
from tests.validation_helpers import sample_candles, validation_settings


def test_stress_tester_generates_scenarios():
    df = sample_candles(180)
    X, y, features = create_features(df, expiration_candles=1)
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)

    result = StressTester(model, df, features, validation_settings()).run()

    assert result["summary"]["total_scenarios"] == 10
    assert any(row["scenario"] == "payout_reduced_10" for row in result["results"])
