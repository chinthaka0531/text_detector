import cv2
import pytesseract as pt
from textclassifier import TextClassifier
from PIL import Image
import numpy as np

# pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configuring the classifier
weights_path = 'weights.pt'
detector = TextClassifier(weights_path)
colors = [[255, 0, 0], [0, 166, 0], [0, 0, 255]]  # Bounding box colors for each class
names = detector.predictor.names


def object_detection(in_file_path):
    original_img = cv2.imread(in_file_path)
    img, prediction = detector.predict(original_img, colors, conf_thresh=0.5, iou_thresh=0.2, filter_classes=None)
    return original_img, img, prediction


def ocr(img, prediction):
    prediction_sort = np.array(prediction)[:, 1:].astype(np.float)
    prediction = np.array(prediction)[prediction_sort[:, 2].argsort()]

    extracted_text = {}
    con_text = ''

    for i, pred in enumerate(prediction):
        [cls_name, cls, xmin, ymin, w, h, conf] = pred
        cls, xmin, ymin, w, h, conf = int(cls), int(xmin), int(ymin), int(w), int(h), float(conf)
        roi = img[ymin:ymin + h, xmin:xmin + w]
        bw_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        (thresh_roi, _) = cv2.threshold(bw_roi, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        pil_img = Image.fromarray(bw_roi)

        area_name = str(i + 1) + ' - ' + cls_name.replace('_', ' ').upper()
        txt = pt.image_to_string(pil_img)
        extracted_text[area_name] = txt
        con_text = con_text + ' ' + txt
    return extracted_text, con_text
