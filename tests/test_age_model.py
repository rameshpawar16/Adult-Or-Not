from src.train import AdultorNot, train_and_save_model
from pathlib import Path
import joblib


def test_model_training():
    model = AdultorNot()
    model.train()

    assert hasattr(model, "predict")

def test_model_prediction():
    model = AdultorNot()
    model.train()

    assert model.predict(18) == "Adult"
    assert model.predict(45) == "Adult"
    assert model.predict(3) == "Minor"
    assert model.predict(17) == "Minor"


def test_model_file_saved(tmp_path):
    model_file = tmp_path / "model.pkl"

    train_and_save_model(model_file)

    assert Path(model_file).exists()

    loaded_model = joblib.load(model_file)
    assert hasattr(loaded_model, "predict")