import numpy as np
import os
import tflite_runtime.interpreter as tflite

from io import BytesIO
from PIL import Image
from urllib import request

MODEL_NAME = os.getenv('MODEL_NAME', 'model_2024_hairstyle_v2.tflite'
                       )
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def predict(url):
    img = download_image(url)
    img = prepare_image(img, (200, 200))

    x = np.array(img, dtype='float32')
    X = np.array([x]) / 255.0

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    
    return float(preds[0, 0])


def lambda_handler(event, context):
    url = event['url']
    print(url)
    pred = predict(url)
    result = {
        'prediction': pred
    }

    return result


# url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


