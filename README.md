# Text Detector

- Here we have used YOLO V7 for detect text regions for a given image.
- After detecting text regions we are feeding cropped bounding boxes to tesseract OCR.
- You can find the full model with the flask app here.

### Installation

- You have to install tesseract in your PC.
- Depending on your OS you can find the instalation guide [here](https://tesseract-ocr.github.io/tessdoc/Installation.html).
- Using following you can clone the project.
````
git clone https://github.com/chinthaka0531/text_detector.git
````
- After that you can cd into the project folder
````
cd text_detector
````
- Now you can clone YOLO V7 into the project folder
````
git clone https://github.com/WongKinYiu/yolov7.git
````
- Now install requirements.txt
````
pip install -r requirements.txt
````
- Pretrained weights can be downloaded from [here](https://drive.google.com/uc?export=download&id=1QQ5S9Du5b-tHMr4FKaCC7O-g1s81qjrJ)
- You can put [weights.pt](https://drive.google.com/uc?export=download&id=1QQ5S9Du5b-tHMr4FKaCC7O-g1s81qjrJ) to the content root of the project.

- Now you can run the flask app.

````
python app.py
````

Thank you
