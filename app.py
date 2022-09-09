from flask import Flask, render_template, request
from dl_model import object_detection, ocr
import cv2
import os

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'files/in')
OUT_PATH = os.path.join(BASE_PATH, 'files/pred')
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('ocr.html', name="")


@app.route('/submitted', methods=['POST', 'GET'])
def profile_ocr():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        out_filename = upload_file.filename
        in_file_path = os.path.join(UPLOAD_PATH, out_filename)
        upload_file.save(in_file_path)

        original_img, plotted_img, prediction = object_detection(in_file_path)
        cv2.imwrite(os.path.join(OUT_PATH, out_filename), plotted_img)
        text_dict, cont_text = ocr(original_img, prediction)
        return render_template('ocr.html', upload=True, upload_image=out_filename, text=text_dict)

    return render_template('ocr.html')


if __name__ == '__main__':
    app.run(debug=True)