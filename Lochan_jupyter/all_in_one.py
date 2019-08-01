# Create a CPU instance for inference, using Mask-RCNN library, make a snapshot of the same

# Create a flask wrapper around them to hit it with images (which will be base64 encoded), than it returns a json
# with bbox and rois

# Implement the below mentioned class write down init fn to use them all

# Test classification for accuracy with different images especially the one based on TF

# Use this class over the jupyter notebook, with the help of matplotlib

import requests, base64, json, skimage.io, cv2
from PIL import Image
from io import BytesIO
import numpy as np

class KYC(object):
    """
    @author Aman Khandelia

    """

    def __init__(self, class_mappings = None, ips = None):
        self.classifiers = ["document_orientation_classifier", "document_classifier"]
        self.extractors = ["aadhaar_text_strip_extraction", "aadhaar_document_extraction" ,"pan_text_strip_extraction", "pan_document_extraction"]
        self.document_extraction_possible = ["aadhaar", "pan"]
        self.text_strip_extraction_possible = ["aadhar_half_front", "pan_front"]
        self.workers = self.classifiers + self.extractors
        self.class_mappings = class_mappings if class_mappings is not None else {worker:{} for worker in self.workers}
        self.ip_dict = ips if ips is not None else {worker:"localhost" for worker in self.workers}
        self.text_strip_ordering = {"pan_front":['Name', 'Pan Number'], "aadhar_half_front":['Name', 'DOB', 'Gender', 'Aadhaar Number']}
        self.IMAGE_PATH_KEYWORD = 'image_path'
        self.TEXT_STRIP_EXTRACTOR_KEYWORD = '_text_strip_extraction'
        self.DOCUMENT_EXTRACTOR_KEYWORD = '_document_extraction'
        self.IMAGE_BASE64_KEYWORD = 'image_base64'

    def _rotate_bound_(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def _encode_file_base64_(self, file_path):
        """
        Read file and encode it to base64 format
        :param file_path: where the file is stored
        :return: base64 string representing the file
        """
        encoded_file = base64.b64encode(open(file_path, 'rb').read())
        return self._base64_to_str(encoded_file)
        # return str(encoded_file)[2:-1]

    def _get_payload_(self, worker, material):
        """

        Data with which a given worker (remotely deployed rest api) can process and give
        the correct response

        :param worker:
        :param material:
        :return:
        """
        # image_path = material[IMAGE_PATH_KEYWORD]
        # encoded_image = base64.b64encode(open(image_path, 'rb').read())
        # payload_dict['image'] = str(encoded_image)[2:-1]
        payload_dict = {}
        if self.IMAGE_BASE64_KEYWORD not in material:
            payload_dict['image'] = self._encode_file_base64_(material[self.IMAGE_PATH_KEYWORD])
        payload_dict['image'] = material[self.IMAGE_BASE64_KEYWORD]
        if worker in self.extractors:
            payload_dict['item_of_interest'] = material['detections_required']
        return payload_dict

    def _parse_response_(self, worker, material, response):
        """

        :param worker:
        :param material:
        :param response:
        :return:
        """

        # DONE: For reorientation of image you have to return the modified base64 value of the image
        # DONE: For classification return just the name of the class
        # DONE: For document extraction, return a dict that will have the class name (i.e. aadhaar_front, pan_front) as key and base64_image as value
        # DONE: For text strip extraction, return a dict with key being metadata label and base64_text_strip as value
        print("print worker")
        print(worker)
        data = response.json()
        print(data)
        print(response)
        print("$$$$$$$$$$$$$$$$$$$$$")
        if worker is self.workers[0]:
            angle_pointer = data['orientation']
            rotation_angle = (360 - (int(angle_pointer) * 90))
            return self._rotate_base64_image(material[self.IMAGE_BASE64_KEYWORD], rotation_angle)
            # return material[self.IMAGE_BASE64_KEYWORD]
        elif worker is self.workers[1]:
            if data['PredictedID'] == 'AAdhar':
                data['PredictedID'] = 'aadhaar'
            elif data['PredictedID'] =='PANCARD':
                data['PredictedID'] = 'pan'

            return data['PredictedID']
        elif self.DOCUMENT_EXTRACTOR_KEYWORD in worker:
            print(str(worker) + "this is the worker")
            #import pdb;pdb.set_trace()
            parsed_response = {}
            masks = data['masks']
            print('Len of data mask is')
            # NOTE: Current approach will only take one mask of particular type into consideration
            # i.e., if there are multiple aadhaar_front, only one will be shown to user
            for mask in masks:
                # print(mask)
                class_id = mask[mask.index('_')+1:mask.rindex('_')]
                parsed_response[self.class_mappings[worker][class_id]] = masks[mask]
            return parsed_response
        elif self.TEXT_STRIP_EXTRACTOR_KEYWORD in worker:
            text_strips = data['rois']
            text_strips = sorted(text_strips, key=lambda x: x[0])
            base64_text_strips = self._rois_to_strips(material[self.IMAGE_BASE64_KEYWORD], text_strips)
            diff = 0
            if len(base64_text_strips) > len(self.text_strip_ordering[material['mask_type']]):
                diff = len(base64_text_strips) - len(self.text_strip_ordering[material['mask_type']])
            extras = [str(i) for i in range(diff)]
            return dict(zip(self.text_strip_ordering[material['mask_type']] + extras, base64_text_strips))
        elif worker in self.workers:
            print("ERROR: {} worker is supported, please implement the functionality for the same".format(worker))
        else:
            print("ERROR: {} worker is invalid, it is not part of the listed workers".format(worker))

    def _rotate_base64_image(self, image, rotation_angle):
        """

        :param image:
        :param rotation_angle:
        :return:
        """
        image_np = self._base64_to_numpy_array_(image)
        rotated_image = self._rotate_bound_(image_np, rotation_angle)
        return self._numpy_array_to_base64_(rotated_image)


    def _rois_to_strips(self, base64_image, rois):
        """

        :param base64_image:
        :param rois:
        :return:
        """
        image = self._base64_to_numpy_array_(base64_image)
        strips = []
        for box in rois:
            strips.append(self._numpy_array_to_base64_(image[box[0]:box[2], box[1]:box[3]]))
        return strips

    def _base64_to_numpy_array_(self, base64_encoded_image):
        """

        :param base64_encoded_image:
        :return:
        """
        imgdata = base64.b64decode(base64_encoded_image)
        img = skimage.io.imread(imgdata, plugin='imageio')
        return img

    def _numpy_array_to_base64_(self, image):
        """

        :param image:
        :return:
        """
        img = Image.fromarray(image)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return self._base64_to_str(img_str)

    def _base64_to_str(self, base64_bytes):
        """

        :param base64_bytes:
        :return:
        """
        # Note: Base64 string contains the extra b and single_quotes which have to be
        # removed before proceeding with decoding otherwise decoding will fail.
        # Hence the slicing.
        return str(base64_bytes)[2:-1]

    def _get_url_(self, worker):
        """

        :param worker:
        :return:
        """
        return 'http://' + self.ip_dict[worker] + '/' + worker
    def _get_required_item_(self, worker, material):
        """

        :param worker:
        :param material:
        :return:
        """
        print(str(worker))
        print( str(material) + "this is the material passed")
        data = self._get_payload_(worker, material)
        print('Expecting response')
        response = requests.post(self._get_url_(worker), json=data)
        # print(type(response))
        return self._parse_response_(worker, material, response)


    def parse_image(self, image_path, classify = True, orientation_correction = False, document_extraction = True, text_strip_extraction = True, class_name = None):
        """

        :param image_path:
        :param classify:
        :param orientation_correction:
        :param document_extraction:
        :param text_strip_extraction:
        :param class_name:
        :return:
        """
	
        response_dict = {}
        base64_image = self._encode_file_base64_(image_path)
        payload = {self.IMAGE_BASE64_KEYWORD: base64_image}
    
        if classify and class_name is None:
            class_name = self._get_required_item_(self.classifiers[1], payload)
            print('Class is :' + str(class_name))
            return
        if orientation_correction and class_name in self.document_extraction_possible:
                payload[self.IMAGE_BASE64_KEYWORD] = self._get_required_item_(self.classifiers[0], payload)
        response_dict['class_name'] = class_name
        print("done")
        #import pdb; pdb.set_trace()
        if class_name in self.document_extraction_possible and document_extraction:
            payload['detections_required'] = ['masks']
            masks = self._get_required_item_(class_name + self.DOCUMENT_EXTRACTOR_KEYWORD, payload)
            response_dict.update({'masks':masks})
            if text_strip_extraction:
                for mask in masks:
                    if mask in self.text_strip_extraction_possible:
                        payload['detections_required'] = ['rois']
                        payload[self.IMAGE_BASE64_KEYWORD] = masks[mask]
                        payload['mask_type'] = mask
                        strips = self._get_required_item_(class_name + self.TEXT_STRIP_EXTRACTOR_KEYWORD, payload)
                        response_dict[mask] = strips
        return response_dict


# image_path = "/home/amank/Documents/datasets/instance_segmentation_vishnu/aadhar1_files/val/57b81a8d934aecea40503bdc.jpg"
# ip_dict = {'document_orientation_classifier':None,
#  'document_classifier':'34.66.104.73',
#  'aadhaar_text_strip_extraction':'35.224.159.52',
#  'aadhaar_document_extraction':'104.154.234.22',
#  'pan_text_strip_extraction':None,
#  'pan_document_extraction': None}
# class_mappings = {'document_orientation_classifier':None,
#  'document_classifier':None,
#  'aadhaar_text_strip_extraction':None,
#  'aadhaar_document_extraction':{'1':"aadhar_half_front", '2':"aadhar_half_back"},
#  'pan_text_strip_extraction':None,
#  'pan_document_extraction': {'1':"pan_front"}}
# kyc_presenter = KYC(class_mappings=class_mappings, ips=ip_dict)
# kyc_presenter.parse_image(image_path, document_extraction=True, text_strip_extraction=True)
