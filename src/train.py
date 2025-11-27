from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

class AdultorNot:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, max_number: int = 100)-> None:
        x = np.arange(max_number).reshape(-1, 1)
        y = (x[:, 0] < 18).astype(int)
        self.model.fit(x, y)

    def predict(self, number: int) -> str:
        pred = self.model.predict([[number]])[0]
        return "Adult" if number >= 18  else "Minor"


def train_and_save_model(model_path: str | Path = "model.pkl") -> Path:
    cls = AdultorNot()
    cls.train()
    model_path = Path(model_path)
    joblib.dump(cls, model_path)
    return model_path