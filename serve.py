from gensim import *
from gensim.models import *
import re
import unicodedata
import os
import codecs
import matplotlib.pyplot as plt
from flask import Flask, jsonify, g, redirect, request, url_for
app = Flask(__name__)

@app.route('/userreview')
def userreview():
	return jsonify({"success":.85})