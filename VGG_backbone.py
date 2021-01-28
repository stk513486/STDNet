# -*- coding: utf-8 -*-



import tensorflow as tf
import numpy as np


class VGG_10():
    
    def __init__(self,vgg_file = "./vgg16.npy" ):

        self.param_dict = np.load(vgg_file , allow_pickle=True ,  encoding='latin1').item()
        
        self.W_1_1 = tf.Variable(self.param_dict["conv1_1"][0])
        self.b_1_1 = tf.Variable(self.param_dict["conv1_1"][1])
        
        self.W_1_2 = tf.Variable(self.param_dict["conv1_2"][0])
        self.b_1_2 = tf.Variable(self.param_dict["conv1_2"][1])
        
        self.W_2_1 = tf.Variable(self.param_dict["conv2_1"][0])
        self.b_2_1 = tf.Variable(self.param_dict["conv2_1"][1])
        
        self.W_2_2 = tf.Variable(self.param_dict["conv2_2"][0])
        self.b_2_2 = tf.Variable(self.param_dict["conv2_2"][1])
        
        self.W_3_1 = tf.Variable(self.param_dict["conv3_1"][0])
        self.b_3_1 = tf.Variable(self.param_dict["conv3_1"][1])

        self.W_3_2 = tf.Variable(self.param_dict["conv3_2"][0])
        self.b_3_2 = tf.Variable(self.param_dict["conv3_2"][1])

        self.W_3_3 = tf.Variable(self.param_dict["conv3_3"][0])
        self.b_3_3 = tf.Variable(self.param_dict["conv3_3"][1])
        
        self.W_4_1 = tf.Variable(self.param_dict["conv4_1"][0])
        self.b_4_1 = tf.Variable(self.param_dict["conv4_1"][1])
        
        self.W_4_2 = tf.Variable(self.param_dict["conv4_2"][0])
        self.b_4_2 = tf.Variable(self.param_dict["conv4_2"][1])
        
        self.W_4_3 = tf.Variable(self.param_dict["conv4_3"][0])
        self.b_4_3 = tf.Variable(self.param_dict["conv4_3"][1])
        

        
    def forward(self , x ):
        
        with tf.variable_scope("VGG_backbone"):
            z = tf.nn.conv2d(x , self.W_1_1 , [1,1,1,1] , padding = "SAME") + self.b_1_1
            z = tf.nn.relu(z)    
            z = tf.nn.conv2d(z , self.W_1_2 , [1,1,1,1] , padding = "SAME") + self.b_1_2
            z = tf.nn.relu(z)
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




            z = tf.nn.conv2d(z , self.W_2_1 , [1,1,1,1] , padding = "SAME") + self.b_2_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_2_2 , [1,1,1,1] , padding = "SAME") + self.b_2_2
            z = tf.nn.relu(z)
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



            z = tf.nn.conv2d(z , self.W_3_1 , [1,1,1,1] , padding = "SAME") + self.b_3_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_3_2 , [1,1,1,1] , padding = "SAME") + self.b_3_2
            z = tf.nn.relu(z)        
            z = tf.nn.conv2d(z , self.W_3_3 , [1,1,1,1] , padding = "SAME") + self.b_3_3
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        
        
            z = tf.nn.conv2d(z , self.W_4_1 , [1,1,1,1] , padding = "SAME") + self.b_4_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_4_2 , [1,1,1,1] , padding = "SAME") + self.b_4_2
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_4_3 , [1,1,1,1] , padding = "SAME") + self.b_4_3
            z = tf.nn.relu(z)

        
            return z
