from flask import Flask, flash, request, redirect, url_for, render_template
#import urllib.request
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
#import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_content():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename, os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        global content
        #content = Image.open(os.path.join(os.getcwd(),app.config['UPLOAD_FOLDER']+'/'+filename))
        content = cv2.imread(os.path.join(os.getcwd(),app.config['UPLOAD_FOLDER']+'/'+filename), cv2.IMREAD_GRAYSCALE)
        #content = image.img_to_array(content, dtype='uint8')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

#copy of method for style
@app.route('/', methods=['POST'])
def upload_style():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filenamejr = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filenamejr))
        print('upload_image filename: ' + filenamejr, os.path.join(app.config['UPLOAD_FOLDER'], filenamejr))
        flash('Image successfully uploaded and displayed below')
        global style
        #content = Image.open(os.path.join(os.getcwd(),app.config['UPLOAD_FOLDER']+'/'+filename))
        style = cv2.imread(os.path.join(os.getcwd(),app.config['UPLOAD_FOLDER']+'/'+filenamejr), cv2.IMREAD_GRAYSCALE)
        #content = image.img_to_array(content, dtype='uint8')
        return render_template('index.html', filename=filenamejr)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

#global img 
#img = cv2.imread('./static/uploads/IMG_3005.jpg', cv2.IMREAD_GRAYSCALE)

@app.route('/GAUSseg/', methods=['GET'])    
def GAUSseg():
    segmented6 = cv2.adaptiveThreshold(content,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2)
    return print(cv2.imshow('',segmented6))

@app.route('/MEANseg/', methods=['GET'])
def MEANseg():
    segmented8 = cv2.adaptiveThreshold(image.img_to_array(content, dtype='uint8'),255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)
    segmented8 = np.array(segmented8)
    return segmented8
    #return print(cv2.imshow('', segmented8))## plan to have optional user inputs for the 
                                             ## maxValue, adaptiveMethod, thresholdType, 
                                             ## blockSize, and C (weighted mean - constant)

@app.route('/maskInter/', methods=['POST', 'GET'])
def maskInter():
    import numpy.ma as ma
    im = MEANseg()
    mask = ma.masked_where(im>0, im)
    segmented_mask = ma.masked_array(im,mask)
    plt.figure(figsize=(7.50,7.50))
    plt.imshow(content, cm.Spectral, interpolation='bilinear')
    #plt.imshow(im, cm.Spectral)
    #plt.imshow(segmented7_mask, 'twilight', interpolation='bilinear', alpha=0.5)
    plt.imshow(segmented_mask, cm.gist_earth, interpolation='bilinear', alpha=0.7)
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'this_plot2.png')
    plt.savefig(filename, dpi=72)
    print(filename)
    return render_template('plotShow.html', filename=filename)

    # here the user will select effects & interpolation type from a drop down bar
    # the alpha value through keyboard/arrows in text box


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()