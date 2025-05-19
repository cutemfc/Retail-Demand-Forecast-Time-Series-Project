import pickle

def load_model(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, input_df):
    return model.predict(input_df)
