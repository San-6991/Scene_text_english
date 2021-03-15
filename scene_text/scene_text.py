# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import torch
import cv2
# import the necessary packages
from easyocr import Reader
import argparse
import cv2
import subprocess
import os


import numpy as np

from flask import Flask
from flask import request
from flask import render_template
import os


app = Flask(__name__)
UPLOAD_FOLDER = os.getcwd()
DEVICE = "cpu"
MODEL = None





def predict(image_location):
    def cleanup_text(text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()
    la = 'en'
    gpu_temp = -1
    # break the input languages into a comma separated list
    langs = la.split(",")
    print("[INFO] OCR'ing with the following languages: {}".format(langs))
    # load the input image from disk
    image = cv2.imread(image_location)
    # OCR the input image using EasyOCR
    print("[INFO] OCR'ing input image...")


    reader = Reader(langs, gpu=gpu_temp > 0)
    results = reader.readtext(image)
    res_temp = reader.readtext(image, detail = 0)
    print('recognition completed')
    text_file = open("sample.txt", "w")
    n = text_file.write(str(res_temp))
    text_file.close()


    # loop over the results
    for (bbox, text, prob) in results:
        # display the OCR'd text and associated probability
        print("[INFO] {:.4f}: {}".format(prob, text))
        # unpack the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        # cleanup the text and draw the box surrounding the text along
        # with the OCR'd text itself
        text = cleanup_text(text)
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, text, (tl[0], tl[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)
    #cv2.imwrite('test4.png',image)
    print(str(res_temp))
    return str(res_temp)


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            #image_file.save(image_location)
            pred = predict(image_location)
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12000, debug=True)
