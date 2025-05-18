import numpy as np
import litserve as ls
import pickle
import pandas as pd
import joblib

class InferenceAPI(ls.LitAPI):
    def setup(self, device = "cpu"):
        # with open("models/full_pipeline.pkl", "rb") as pkl:
        #     self._model = pickle.load(pkl)
        self._model = joblib.load("models/full_pipeline.pkl")

    def decode_request(self, request):
        try:
            columns = request['columns']
            rows = request['rows']
            df = pd.DataFrame(rows, columns=columns)
            print(df)
            return df
        except:
            return None

    def predict(self, x):
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        response = {
            "message": message,
            "data": output.tolist(),
        }
        return response