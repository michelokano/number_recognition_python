from flask import render_template, request, redirect, session, flash, url_for, send_from_directory
from image_processing import number_recognition
from delete_after_images import delete
from main import app

@app.route('/')
def index():
    delete()
    # list = []
    list = number_recognition()
    return render_template('index.html', list=list)

@app.route('/after_images/<filename>')
def image(filename):
    return send_from_directory(('after_images'), filename)