import inspect
import os
from scipy import ndimage, misc

import numpy as np
import tensorflow as tf
import time
import base64, json
import pickle as pkl
from flask import Flask, request, abort, jsonify
import skimage.io

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None, num_classes=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join("/home/ofsdms/Lochan/doc-classifier-checkpoint/", "vgg16.npy")
            vgg16_npy_path = path
            print(path)
        # this data_dict contain all the s
        
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.num_classes = num_classes
        self.data_dict_shape = {}
        print([value[1].shape for value in self.data_dict.values()])
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        start_time = time.time()
        print("build model started")

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        # assert tf.shape(red)[1:] == [224, 224, 1]
        # assert tf.shape(green)[1:] == [224, 224, 1]
        # assert tf.shape(blue)[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            tf.subtract(tf.cast(blue, dtype=tf.float32), VGG_MEAN[0]),
            tf.subtract(tf.cast(green, dtype=tf.float32), VGG_MEAN[1]),
            tf.subtract(tf.cast(red, dtype=tf.float32), VGG_MEAN[2]),
        ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        # self.bgr = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
        self.bgr = bgr
        self.conv1_1 = self.conv_layer(self.bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.pool6 = tf.reduce_mean(self.pool5, [1, 2])

        self.fc6 = self.fc_layer(self.pool6, "fc6")
        # assert self.fc6.get_shape().as_list()[1:] == [4096]
        # self.relu6 = tf.nn.relu(self.fc6)
        #
        # self.fc7 = self.fc_layer(self.relu6, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)
        #
        # self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc6, name="prob")

        self.data_dict = None


        print(("build model finished: %ds" % (time.time() - start_time)))

        return self

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            # shape = tf.shape(bottom)
            # dim = 1
            # for d in shape[1:]:
            #     dim *= d
            # x = tf.reshape(bottom, [-1, dim])
            # shape_of_bottom = tf.shape(bottom)
            # if name == 'fc6':
            #     x = tf.reshape(bottom, [-1, shape_of_bottom[1]*shape_of_bottom[2]*shape_of_bottom[3]])
            # else:
            x = bottom
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_conv_filter(self, name):
        return tf.Variable(initial_value=self.data_dict[name][0], name='filter')
        # return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        if self.num_classes is not None and name == 'fc6':
            return tf.get_variable(name='biases', shape=(self.num_classes), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        return tf.Variable(initial_value=self.data_dict[name][1], name='biases')

        # return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        if self.num_classes is not None and name == 'fc6':
            return tf.get_variable(name='weights', shape=(512, self.num_classes), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        return tf.Variable(initial_value=self.data_dict[name][0], name='weights')
        # return tf.constant(self.data_dict[name][0], name="weights")
    def get_loss(self, labels):
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.fc6)
        return self
    def calculate_accuracy(self, labels):
        predictions = tf.argmax(self.prob, axis=1)
        equality = tf.equal(predictions, labels)
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))


def predict(image, sess):
    session = sess
    image = ((misc.imresize(image, (image_size, image_size)).astype(float)).astype(np.float32))
    feedDict = {images:[image]}
    predCls, probVal = session.run([predictions, model.prob], feed_dict=feedDict)
    score = round(probVal[0][predCls[0]] * 100, 2)
    predId = labelNumDict[predCls[0]]
    if score < minimumScore:
        print(predId)
        predId = 'Others'
    xLabel = "Predicted ID: {} with confidence {} %".format(predId, score)
    print(xLabel)
    del image, feedDict, predCls, probVal
    return {"PredictedID": predId, "Confidence": score}

app = Flask(__name__)
image_size = 244
minimumScore = 80.00
labelNumDict = pkl.load(open('/home/ofsdms/Lochan/doc-classifier-checkpoint/classDict.pkl', 'rb'))
vgg16_weight_file_path, path_to_model = 'vgg16.npy', '/home/ofsdms/Lochan/doc-classifier-checkpoint/model.ckpt-99'
model = Vgg16(vgg16_npy_path=vgg16_weight_file_path, num_classes=5)
images = tf.placeholder(name="placeholder_image", shape=(None, 244, 244, 3), dtype=tf.float32)
model.build(images)
predictions = tf.argmax(model.prob, axis=1)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, path_to_model)
def base64_to_skimage(base64_encoded_image):
    # TODO: Ensure that the b and quotes have been removed from the string-fied image encoding
    # if isinstance(base64_encoded_image, bytes):
    #     base64_encoded_image = base64_encoded_image.decode("utf-8")
    # Note: Image string might contain the extra b and single_quotes which have to be
    # removed before proceding with decoding otherwise decoding will fail
    imgdata = base64.b64decode(base64_encoded_image)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img

@app.route('/document_classifier', methods=['POST'])
def get_detections():
    if not request.json or not 'image' in request.json:
        abort(400)
    image = base64_to_skimage(request.json['image'])
    output = predict(image, sess)
    return jsonify(output), 201


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8083, debug=True)            