from flask import Flask, render_template

application = Flask(__name__, '/static')

@application.route('/')
def home() -> str:
    return render_template('home.html')