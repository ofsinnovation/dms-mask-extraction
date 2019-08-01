from flask import Flask, request,jsonify
from tensorflow.python.keras.models import load_model
import numpy as np
from scipy import misc
import base64
import skimage.io

# Initialize the Flask application
app = Flask(__name__)

# model_path = '/home/madhevan/Hybrid_Aadhar_Pan_Orientation_Classfier_vgg16.h5'
model_path = 'weight_file.h5'
orientation_model = load_model(model_path)
image_size = 224
dict_map = {0: 'FlipLeft',1: 'FlipRight', 2 : 'Inverted', 3: 'Normal' }
#Dictionary to rotate as multiples of 90 degrees
degree_map = {0: '1',1: '3', 2 : '2', 3: '0' }

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
    image = (misc.imresize(image_to_process, (image_size, image_size)).astype(float)).astype(float)
    image = np.reshape(image, (1, image_size, image_size, -1))
    prediction = model.predict(image, verbose = 0)
    prediction = np.argmax(prediction, axis=1)
    print(prediction)
    return class_dict[prediction[0]]

    #return prediction[0]

# route http posts to this method
@app.route('/document_orientation_classifier', methods=['POST'])
def find_orientation():
    image_64_encode = request.json['image']
    im = base64_to_skimage(image_64_encode)
    orientation_image = test_model_single_image(None, im, degree_map, orientation_model)
    # build a response dict to send back to client
    response = {'orientation': '{}'.format(orientation_image)}
    return jsonify(response)


# start flask app
app.run(host="0.0.0.0", port=80, debug=True)