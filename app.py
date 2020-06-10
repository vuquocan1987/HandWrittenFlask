from flask import Flask

from flask import render_template,url_for
from flask import request, redirect
from werkzeug.utils import secure_filename
from datetime import datetime

import cv2
from src.util import HandWrittingPredictor
import numpy as np
import tensorflow as tf
import os
import re

app = Flask(__name__)
predictor = HandWrittingPredictor()
app.config["IMAGE_UPLOADS"] = os.path.join('static','upload')
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG","JPG","PNG","GIF"]
app.config["MAX_CONTENT_LENGTH"] = 50*1024*1024

def allowed_image(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".",1)[1]
    return ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]
def allowed_image_filesize(filesize):
    return int(filesize) <= app.config["MAX_CONTENT_LENGTH"] 

@app.route("/")
def home():
    return render_template("public/upload_image.html")

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content
def load_and_predict(path):
    img = cv2.imread(path)
    predicts = predictor.predict(img)
    return predicts[0]

@app.route("/files",methods=["GET"])
def get_image():
    names = os.listdir(app.config["IMAGE_UPLOADS"])
    image_paths = [os.path.join(app.config["IMAGE_UPLOADS"],name) for name in names]

    return render_template('public/all_image.html', image_paths=image_paths)

@app.route("/classification",methods = ["GET","POST"])
def upload_image():
#   test_model()
#    model.summary()
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.filename == "":
                print("No filename")
                return redirect(request.url)
            if "filesize" in request.cookies:
                if not allowed_image_filesize(request.cookies["filesize"]):
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url)
            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                path = os.path.join(app.config["IMAGE_UPLOADS"],filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"],filename))
                predicts = load_and_predict(path)
            
            return render_template("public/show_result.html",predicts = predicts)
    return render_template("public/upload_image.html")

