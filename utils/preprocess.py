import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(df, encoders=None):
    if encoders is None:
        encoders = {}
        for col in ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]:
            df[col] = encoders[col].transform(df[col])
    return df, encoders