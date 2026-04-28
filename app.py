from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("placement_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    cgpa = float(request.form["cgpa"])
    aptitude = float(request.form["aptitude"])
    communication = float(request.form["communication"])
    internships = float(request.form["internships"])
    projects = float(request.form["projects"])

    data = np.array([[cgpa, aptitude, communication, internships, projects]])

    result = model.predict(data)[0]

    if result == 1:
        output = "🎉 Likely to be Placed"
        advice = "Great profile! Keep improving coding skills, aptitude, communication, and mock interviews."

    else:
        output = "📈 Needs Improvement"
        advice = "Build projects, improve aptitude daily, strengthen communication, and maintain good CGPA."

    return render_template(
        "index.html",
        prediction_text=output,
        advice=advice
    )


if __name__ == "__main__":
    app.run(debug=True)