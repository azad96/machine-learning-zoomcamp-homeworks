import pickle

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_pickle('model1.bin')
dv = load_pickle('dv.bin')

client = {"job": "management", "duration": 400, "poutcome": "success"}

X = dv.transform(client)
y_pred = model.predict_proba(X)[:, 1]

print(y_pred)