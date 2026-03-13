from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# load model
model = pickle.load(open("sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def index():

    prediction = ""

    if request.method == "POST":

        review = request.form["review"]

        review_vec = vectorizer.transform([review])

        result = model.predict(review_vec)

        prediction = result[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
