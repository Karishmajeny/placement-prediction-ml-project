import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("dataset.csv")

X = data[["cgpa","aptitude","communication","internships","projects"]]
y = data["placed"]

model = LogisticRegression()
model.fit(X,y)

pickle.dump(model, open("placement_model.pkl","wb"))

print("Model Trained Successfully")