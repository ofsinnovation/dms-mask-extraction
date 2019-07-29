from flask import Flask, request, Response,jsonify
#import jsonpickle
import numpy as np
import cv2
import re
from tensorflow.python.keras.models import load_model
from os import listdir
import cv2
import numpy as np
from scipy import ndimage, misc
import os
from io import BytesIO
from PIL import Image
import base64
import skimage.io
# Initialize the Flask application
app = Flask(__name__)

model_path = '/home/ofsdms/Hybrid_Aadhar_Pan_Orientation_Classfier_vgg16.h5'
orientation_model = load_model(model_path)
orientation_model._make_predict_function()
image_size = 224

def base64_to_skimage(base64_encoded_image):
    # TODO: Ensure that the b and quotes have been removed from the string-fied image encoding
    # if isinstance(base64_encoded_image, bytes):
    #     base64_encoded_image = base64_encoded_image.decode("utf-8")
    # Note: Image string might contain the extra b and single_quotes which have to be
    # removed before proceding with decoding otherwise decoding will fail
    imgdata = base64.b64decode(base64_encoded_image)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img

def test_model_single_image(path_to_model, image_to_process, class_dict = {}, model = None):
    """
    :param path_to_test_set:
    :param path_to_model:
    :return:
    """
    global image_size, max_pixel
    if model is None:
        model = load_model(path_to_model)
    #image1 = ndimage.imread(path_to_image, mode='RGB').astype(np.float)
    image = (misc.imresize(image_to_process, (image_size, image_size)).astype(float)).astype(float)
    image = np.reshape(image, (1, image_size, image_size, -1))
    prediction = model.predict(image, verbose = 0)
    prediction = np.argmax(prediction, axis=1)
    print(prediction)
    return class_dict[prediction[0]]

    #return prediction[0]
dict_map = {0: 'FlipLeft',1: 'FlipRight', 2 : 'Inverted', 3: 'Normal' }
degree_map = {0: '1',1: '3', 2 : '2', 3: '0' }

image_size = 224
# route http posts to this method
@app.route('/document_orientation_classifier', methods=['POST'])
def find_orientation():
    # = request
    # convert string of image data to uint8
    #nparr = np.fromstring(r.data, np.uint8)
    # decode image
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #image_data = re.sub('^data:image/.+;base64,', '', request.form['data'])
    #im = Image.open(base64.b64decode(request.json['encoded_image']))
    #im = Image.open(BytesIO(base64.b64decode(encoded_string)))
    #im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    image_64_encode = request.json['image']
    #image_64_decoded = base64.b64decode(image_64_encode)
    im = base64_to_skimage(image_64_encode)
    cv2.imwrite('/home/ofsdms/Decoded_image_from_json.jpg', im)
    #image = open(image_path, 'rb')
    #image_result = open('/home/madhevan/base_60_decoded.jpg', 'wb')
    #image_result.write(image_64_decode)
    #im = image_64_decoded
    orientation_image = test_model_single_image(None, im, degree_map, orientation_model)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'orientation': '{}'.format(orientation_image)}
    # encode response using jsonpickle
    #response_pickled = jsonpickle.encode(response)

    #return Response(response=response_pickled, status=200, mimetype="application/json")
    return jsonify(response)


# start flask app
app.run(host="0.0.0.0", port=8082)