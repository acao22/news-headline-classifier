# model_logreg.py
import os
import joblib

class Model:
    def __init__(self, weights_path=None):
        here = os.path.dirname(os.path.abspath(__file__))

        vec_path = os.path.join(here, "logreg_vectorizer.joblib")
        mdl_path = os.path.join(here, "logreg_model.joblib")

        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(mdl_path)

    def predict(self, batch):
        X_vec = self.vectorizer.transform(batch)
        return self.model.predict(X_vec).tolist()
