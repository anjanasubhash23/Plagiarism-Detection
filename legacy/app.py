from flask import Flask, render_template, request
from model import Calculate_Percentage
app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("create.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
    data = []
    for key, value in result.items():
        data.append(value)
    c = Calculate_Percentage(data[0], data[1])
    print(c)
    return render_template("result.html", result=f"{c:0,.2f}", text1=data[0], text2=data[1])


if __name__ == "__main__":
    app.debug = True
    app.run()
