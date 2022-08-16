import numpy as np
from collections import OrderedDict
from src.functions import *
from src.layers import *

class VGGNet:
    def __init__(self, input_dim=(1, 28, 28), conv_params={
        'filter_size': 3,
        'pad': 1,
        'stride': 1
    },output_size = 10, weight_init_std=0.01):
        filter_size = conv_params['filter_size']
        filter_pad = conv_params['pad']
        filter_stride = conv_params['stride']
        input_size = input_dim[1]

        #conv_output_size = (input_size - filter_size + 2*filter_pad)/filter_stride + 1
        #pool_output_size = int(filter_num*(conv_output_size/2)*(conv_output_size/2))

        #saving params
        self.params = {}

        #conv1/filter_num = 32
        conv1_filter_num = 32
        self.params['conv1_W'] = weight_init_std*np.random.randn(conv1_filter_num, input_dim[0], filter_size, filter_size)  
        self.params['conv1_b'] = np.zeros(conv1_filter_num)

        #conv2/filter_num = 32
        conv2_filter_num = 32
        self.params['conv2_W'] = weight_init_std*np.random.randn(conv2_filter_num, conv1_filter_num, filter_size, filter_size)
        self.params['conv2_b'] = np.zeros(conv2_filter_num)

        #conv3/filter_num = 32
        conv3_filter_num = 32
        self.params['conv3_W'] = weight_init_std*np.random.randn(conv3_filter_num, conv2_filter_num, filter_size, filter_size)
        self.params['conv3_b'] = np.zeros(conv3_filter_num)

        #conv4/filter_num = 64
        conv4_filter_num = 64
        self.params['conv4_W'] = weight_init_std*np.random.randn(conv4_filter_num, conv3_filter_num, filter_size, filter_size)
        self.params['conv4_b'] = np.zeros(conv4_filter_num)

        #conv5/filter_num = 64
        conv5_filter_num = 64
        self.params['conv5_W'] = weight_init_std*np.random.randn(conv5_filter_num, conv4_filter_num, filter_size, filter_size)
        self.params['conv5_b'] = np.zeros(conv5_filter_num)

        #FCL1/
        self.params['fcl1_W'] = weight_init_std*np.random.randn(conv5_filter_num*7*7, 128)
        self.params['fcl1_b'] = np.zeros(128)

        #FCL2/
        self.params['fcl2_W'] = weight_init_std*np.random.randn(128, output_size)
        self.params['fcl2_b'] = np.zeros(output_size)


        #save_layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['conv1_W'],
                                           self.params['conv1_b'],
                                           conv_params['stride'],
                                           conv_params['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        #(1, 28, 28) -> (conv1_filter_num, 14, 14)

        self.layers['Conv2'] = Convolution(self.params['conv2_W'],
                                           self.params['conv2_b'],
                                           conv_params['stride'],
                                           conv_params['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        #(conv1_filter_num, 14, 14) -> (conv2_filter_num, 7, 7)

        self.layers['Conv3'] = Convolution(self.params['conv3_W'],
                                           self.params['conv3_b'],
                                           conv_params['stride'],
                                           conv_params['pad'])
        self.layers['Relu3'] = Relu()
        #(conv2_filter_num, 7, 7) -> (conv3_filter_num, 7, 7)

        self.layers['Conv4'] = Convolution(self.params['conv4_W'],
                                           self.params['conv4_b'],
                                           conv_params['stride'],
                                           conv_params['pad'])
        self.layers['Relu4'] = Relu()
        #(conv3_filter_num, 7, 7) -> (conv4_filter_num, 7, 7)

        self.layers['Conv5'] = Convolution(self.params['conv5_W'],
                                           self.params['conv5_b'],
                                           conv_params['stride'],
                                           conv_params['pad'])
        self.layers['Relu5'] = Relu()
        #(conv4_filter_num, 7, 7) -> (conv5_filter_num, 7, 7)

        self.layers['Affine1'] = Affine(self.params['fcl1_W'], self.params['fcl1_b'])
        self.layers['Relu6'] = Relu()
        self.layers['Affine2'] = Affine(self.params['fcl2_W'], self.params['fcl2_b'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)

        return np.sum(y == t) / float(x.shape[0])

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['conv1_W'] = self.layers['Conv1'].dW
        grads['conv1_b'] = self.layers['Conv1'].db
        grads['conv2_W'] = self.layers['Conv2'].dW
        grads['conv2_b'] = self.layers['Conv2'].db
        grads['conv3_W'] = self.layers['Conv3'].dW
        grads['conv3_b'] = self.layers['Conv3'].db
        grads['conv4_W'] = self.layers['Conv4'].dW
        grads['conv4_b'] = self.layers['Conv4'].db
        grads['conv5_W'] = self.layers['Conv5'].dW
        grads['conv5_b'] = self.layers['Conv5'].db
        
        grads['fcl1_W'] = self.layers['Affine1'].dW
        grads['fcl1_b'] = self.layers['Affine1'].db
        grads['fcl2_W'] = self.layers['Affine2'].dW
        grads['fcl2_b'] = self.layers['Affine2'].db

        return grads