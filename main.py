import flask 
import os
import imghdr

from flask import request, render_template, abort, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
from utils import border
from flask_dropzone import Dropzone
'''
OTHER POSSIBLE IMPORTS 
-------------- from flask -> Flask,request,jsonify
from flask_cors import CORS
from time import time
from base64 import b64encode
import cv2
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
from matplotlib import pyplot as plt
from utils import parse_cfg, predict_and_save
from darknet import Darknet

'''
### Some constants 
# PROJECT_PATH = '/'


app = flask.Flask(__name__)

app.config["INPUT_PATH"] = "static/input/"
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']

# app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
# app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
# app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
# app.config['DROPZONE_REDIRECT_VIEW'] = 'static/input/'

def validate_image(stream):
	header = stream.read(512)
	stream.seek(0)
	format = imghdr.what(None,header)
	if not format:
		return None
	return '.' +  (format if format != 'jpeg' else 'jpg')

@app.route("/")
def home():
	# return "Hello World"
	return render_template("home.html")

@app.route("/about")
def about():
	# return "Hello Bravo"
	return render_template("about.html")

@app.route('/upload')
def upload():
	files = os.listdir(app.config["INPUT_PATH"])
	files = ["".join(os.path.join("../static/input/",element).split()) for element in files]
	if files:
		return render_template("upload.html",files=files)
	else:
		return render_template("init_upload.html")

@app.route('/upload', methods=['POST'])
def upload_files():
	if request.method == "POST":
		input_file = request.files['file']
		print(input_file,input_file.filename)
		filename = secure_filename(input_file.filename)
		if filename != '':
			# check the extension validity 
			file_ext = os.path.splitext(filename)[-1]
			if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(input_file.stream):
				return "Invalid Image", 400
			input_file.save(app.config["INPUT_PATH"] + filename)
			print("fsdafsfafasfdasf ",input_file.filename)
			border(app.config["INPUT_PATH"] + filename)
	return redirect(url_for("upload"))

@app.route('/upload/<filename>')
def upload_path(filename):
	return send_from_directory(app.config["INPUT_PATH"], filename)


@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

if __name__=="__main__":
	app.run(debug=True, host='127.0.0.1',port=4041)
