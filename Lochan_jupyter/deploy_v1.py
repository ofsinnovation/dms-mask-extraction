from flask import Flask, request, abort, jsonify, app
import base64, tensorflow as tf
import skimage.io


app = Flask(__name__)

from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import model as modellib, utils
from object_localisation import perspective_transform
class InferenceConfig(Config):
    # NOTE: For Cloud deployment
    # Change the CLASS_MAPPING, and NUM_CLASSES as per the weight file for
    # for flawless deployment


    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # Give the configuration a recognizable name
    NAME = "balloon"

    # Number of classes (including background)
    # NUM_CLASSES = 5  # Background + dataset_size
    # For PAN object extraction
    NUM_CLASSES = 2  # Background + [pan_front]

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Class dict for mapping the class name to integer
    # For aadhaar_object_extraction
    # CLASS_MAPPING = {"aadhar_full_front": 1, "aadhar_full_back": 2, "aadhar_half_front": 3, "aadhar_half_back": 4}

    # For pan_object_extraction
    CLASS_MAPPING = {"pan_front": 1}
    # for Cloud
    DEFAULT_LOG_FILE = '/home/ofsdms/logs'
    DEFAULT_WEIGHT_FILE_PATH = '/home/ofsdms/weight_file.h5'
     
    # For aadhaar_object_extraction
    # DEFAULT_API_NAME = '/aadhaar_extraction'

    # For pan_object_extraction
    DEFAULT_API_NAME = '/pan_extraction'

    # for running locally
    # DEFAULT_LOG_FILE = '/home/amank/Documents/logs_file/logs_lochan'
    # DEFAULT_WEIGHT_FILE_PATH = '/home/amank/Documents/weight_files/instance_segmentation/aadhaar_model_v1.h5'
    # DEFAULT_API_NAME = '/aadhaar_extraction'

    # ITEM_OF_INTEREST will be prediction item returned by the API
    # Except for 'masks', everthing else returns a numpy array converted to list
    # In case of the mask, all the images extracted based on the masks by the algorithm are returned in base64 format
    ITEM_OF_INTEREST = 'masks'
    DEVICE_NAME = '/cpu:0'

config_file = InferenceConfig()
transformer = perspective_transform.transform()
with tf.device(config_file.DEVICE_NAME):
    model = modellib.MaskRCNN(mode="inference", config=config_file,
                              model_dir=config_file.DEFAULT_LOG_FILE)
    model.load_weights(config_file.DEFAULT_WEIGHT_FILE_PATH, by_name=True)
    model.keras_model._make_predict_function()

def base64_to_skimage(base64_encoded_image):
    # TODO: Ensure that the b and quotes have been removed from the string-fied image encoding
    # if isinstance(base64_encoded_image, bytes):
    #     base64_encoded_image = base64_encoded_image.decode("utf-8")
    # Note: Image string might contain the extra b and single_quotes which have to be
    # removed before proceding with decoding otherwise decoding will fail
    imgdata = base64.b64decode(base64_encoded_image)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img

def get_item_of_interest(detections, items_required, image):
    items = ('rois', 'masks', 'scores', 'class_ids')
    response_dict = {}
    if items_required is None:
        items_required = config_file.ITEM_OF_INTEREST
        if items_required not in items:
            print("{} is required, which is not part of {}, hence aborting the process".format(items_required, items))
            return "Error: Unable to process the image", 500
        if items_required is not 'masks':
            response_dict[items_required] = detections[items_required].tolist()
        else:
            response_dict[items_required] = transformer.extract_relevant_portion_from_mask(image, detections)
    else:
        for item in items_required:
            if item not in items:
                print("{} is required, which is not part of {}, hence aborting the process".format(item, items))
                return "Error: Unable to process the image", 500
            if item != 'masks':
                response_dict[item] = detections[item].tolist()
            else:
                response_dict[item] = transformer.extract_relevant_portion_from_mask(image, detections)
    return jsonify(response_dict), 201
    # TODO(amank): Find a way to deal with masks, will mostly depend upon the kind the problem we have at hand


@app.route(config_file.DEFAULT_API_NAME, methods=['POST'])
def get_detections():
    if not request.json or not 'image' in request.json:
        abort(400)
    image = base64_to_skimage(request.json['image'])
    items_of_interest = None
    if 'item_of_interest' in request.json:
        items_of_interest =  request.json['item_of_interest']
    detections = model.detect([image])[0]
    # NOTE: detections contains masks, scores, rois and class_id.
    # Out of these it is cost-effecient to send the scores, rois, class_id over the network
    # But in case you wish to use the mask, find a more efficient way to deal with it.
    # It is humungous and will take forever to transmit over a network
    return get_item_of_interest(detections, items_of_interest, image)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)
