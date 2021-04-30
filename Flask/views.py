"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import *
from stocks_movement import app
from .backend import *
import requests

@app.route('/')
def home():
    if request.method == "GET":
        return render_template('search.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
        if request.method == "POST":
            ticker = request.form["company"]
            results = stock_movement(ticker)
        return render_template("search.html", results=results)

