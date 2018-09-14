#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

class Network(object):
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i+1], SigmoidActivator()))

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, samples, labels, rate, epoch):
        for i in range(epoch):
            for d in range(len(samples)):
                self.train_one_sample(samples[d], labels[d], rate)

    def train_on_sample(self, sample, label, rate):
        self.predict(sample)
        pass

    def calc_gradient(self, label):
        output_layer = self.layers[-1]
        delta = output_layer.activator.backward(output_layer.output) * (label - output_layer.output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, self.input) + self.b)

    def backward(self, delta_array):
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_gradient = np.dot(delta_array, self.input.T)
        self.b_gradient = delta_array

    def update(self, learning_rate):
        self.W += learning_rate * self.W_gradient
        self.b += learning_rate * self.b_gradient

class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)