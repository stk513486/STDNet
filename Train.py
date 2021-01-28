# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io as scio
import scipy
from PIL import Image
import time

from VGG_backbone import VGG_10
import tools





def resize_and_adjust_channel(x ,resize_shape = None ,channel = None , Name = None):

    with tf.variable_scope("Adjust_Channel_" + Name):    

        x_resize = tf.compat.v1.image.resize_bilinear(x , resize_shape)

        z = tf.layers.conv2d(x_resize , channel , [1,1] , strides=[1,1] , padding="SAME")

        return z


def output_adjust(x ,resize = None ,channel = None , Name = None):

    with tf.variable_scope("Adjust_Channel_" + Name):    

        z = tf.layers.conv2d(x , channel , [1,1] , strides=[1,1] , padding="SAME")
        
        z_resize = tf.compat.v1.image.resize_bilinear(z , resize)

        return z_resize



def common_conv2d(z , in_filter = None , out_filter = None ,Name = None):

    with tf.variable_scope(Name):    
        W = tf.compat.v1.get_variable(name = Name+"_W" , shape = [1,1,in_filter,out_filter])
        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())
    
        z = tf.compat.v1.nn.conv2d(z , W , strides=[1,1,1,1] ,  padding="SAME") + b
        z = tf.nn.relu(z)    
        
        return z
    
    
def dilated_conv2d(z , in_filter = None , out_filter = None , dilated_rate = None,Name = None):

    with tf.variable_scope(Name):    
        W = tf.compat.v1.get_variable(name = Name+"_W" , shape = [3,3,in_filter,out_filter])
        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())
    
        z = tf.nn.atrous_conv2d(z , W , rate = dilated_rate , padding="SAME") + b
        z = tf.nn.relu(z)    
        
        return z

    
def common_conv3d(z , in_filter = None , out_filter = None , Name = None):
    
    with tf.variable_scope(Name):    
        W = tf.compat.v1.get_variable(name = Name+"_W" , shape = [1,1,1,in_filter,out_filter])
#        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())
    
        z = tf.compat.v1.nn.conv3d(z , W ,strides=[1,1,1,1,1] , padding="SAME")
        z = tf.nn.relu(z)    
        return z
    
    
def dilated_conv3d(z , in_filter = None , out_filter = None , dilated_rate = None , Name=None):
    with tf.variable_scope(Name):    
        
        zero = tf.zeros([1,1,1,in_filter , out_filter])
        W1 = tf.compat.v1.get_variable(name = Name+"_W1" , shape = [1,1,1,in_filter,out_filter])
        W2 = tf.compat.v1.get_variable(name = Name+"_W2" , shape = [1,1,1,in_filter,out_filter])
        W3 = tf.compat.v1.get_variable(name = Name+"_W3" , shape = [1,1,1,in_filter,out_filter])
#        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())
    
        if dilated_rate == 1:
            W = tf.concat([W1,W2,W3] , axis=0)
        elif dilated_rate == 2 :
            W = tf.concat([W1,zero,W2,zero,W3] , axis=0)           
        elif dilated_rate == 3 :
            W = tf.concat([W1,zero,zero,W2,zero,zero,W3] , axis=0)
            
        z = tf.compat.v1.nn.conv3d(z , W ,strides=[1,1,1,1,1] , padding="SAME")
        z = tf.nn.relu(z)    
        return z


def Dense_Spatial_Block(x , name = None):
    
    channel_list = [64,64,64]
    with tf.variable_scope("Dense_Spatial_Block" + "_" + name):
        
        z1 = common_conv2d(x , 512 , 256 , "DSB_1-1")
        z1 = dilated_conv2d(z1 , 256 , channel_list[0] , 1 ,"DSB_1-2")
        
        z2 = tf.concat([x,z1],axis=3)
        z2 = common_conv2d(z2 , 512+channel_list[0] , 256 , "DSB_2-1")
        z2 = dilated_conv2d(z2 , 256 , channel_list[1] , 2 , "DSB_2-2")
        
        z3 = tf.concat([x,z1,z2],axis=3)
        z3 = common_conv2d(z3 , 512+channel_list[0]+channel_list[1], 256 ,"DSB_3-1")
        z3 = dilated_conv2d(z3 , 256 , channel_list[2] , 3 ,"DSB_3-2")
        
        z4 = tf.concat([x,z1,z2,z3],axis=3)
        z4 = common_conv2d(z4 , 512+channel_list[0]+channel_list[1]+channel_list[2] , 512 , "DSB_4-1")
        
        return z4


def Spatial_Channel_Aware_Block(x , name= None):
    with tf.variable_scope("Spatial_Channel_Aware_Block" + "_" + name):
        
        gap = tf.reduce_mean(x , axis=[1,2],keep_dims=True)
        gap = tf.reshape(gap , [tf.shape(x)[0] , tf.shape(x)[3]])
        
        weight = tf.layers.dense(gap , 128)
        weight = tf.nn.relu(weight)
        weight = tf.layers.dense(weight , 512)
        weight = tf.nn.sigmoid(weight)
        
        a = tf.reshape(weight,[tf.shape(x)[0] , 1 , 1 , tf.shape(x)[3]])
        
        z = tf.multiply(a,x)

        return z
    


def Dense_Temporal_Block(x , name = None):
    
    channel_list = [64,64,64]
    with tf.variable_scope("Dense_Temporal_Block" + "_" + name):
        z1 = common_conv3d(x , 512 , 256 , "DTB_1-1")
        z1 = dilated_conv3d(z1 , 256 , channel_list[0] , 1 ,"DTB_1-2")
        
        z2 = tf.concat([x,z1],axis=4)
        z2 = common_conv3d(z2 , 512+channel_list[0] , 256 , "DTB_2-1")
        z2 = dilated_conv3d(z2 , 256 , channel_list[1] , 2 , "DTB_2-2")
        
        z3 = tf.concat([x,z1,z2],axis=4)
        z3 = common_conv3d(z3 , 512+channel_list[0]+channel_list[1] , 256 ,"DTB_3-1")
        z3 = dilated_conv3d(z3 , 256 , channel_list[2] , 3 ,"DTB_3-2")
        
        z4 = tf.concat([x,z1,z2,z3],axis=4)
        z4 = common_conv3d(z4 , 512+channel_list[0]+channel_list[1]+channel_list[2] , 512 , "DTB_4-1")
        
        return z4


    
def Temporal_Channel_Aware_Block(x , name = None):
    
    with tf.variable_scope("Temporal_Channel_Aware_Block" + "_" + name):
        
        gap = tf.reduce_mean(x , axis = [1,2,3],keep_dims=True)
        gap = tf.reshape(gap , [tf.shape(x)[0] , tf.shape(x)[4]])
        
        weight = tf.layers.dense(gap , 128)
        weight = tf.nn.relu(weight)
        weight = tf.layers.dense(weight , 512)
        weight = tf.nn.sigmoid(weight)
        
        a = tf.reshape(weight,[tf.shape(x)[0] , 1 , 1 , 1 , tf.shape(x)[4]])
        
        z = tf.multiply(a,x)
    
        return z
        
    


def Training(batch_SIZE = None, time_step = None, Epoch = None, lr = None):

    
    X_train , Y_train , X_test , Y_test = tools.DataLoader(data_aug = True)


    VGG = VGG_10()

    x = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [batch_SIZE ,time_step , 158 , 238 , 1])
    y = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [batch_SIZE ,time_step , 158 , 238 , 1])
    
    x_reshape = tf.reshape(x , [-1 , 158 , 238 , 1])
    
    LR = tf.compat.v1.placeholder(tf.float32)



    z = resize_and_adjust_channel(x_reshape , [316,476] , 3, "Start")    
    z = VGG.forward(z)

    
    S_1 = Dense_Spatial_Block(z , "DSB_1")
    S_1 = Spatial_Channel_Aware_Block(S_1 , "SCA_1")    
    z_1 = S_1

    z = tf.reshape(z_1 , [batch_SIZE , time_step , tf.shape(z)[1] , tf.shape(z)[2] , tf.shape(z)[3] ],name="Reshape_S_T")

    T_1 = Dense_Temporal_Block(z , "DTB_1")
    T_1 = Temporal_Channel_Aware_Block(T_1 , "TCA_1")    
    z_1 = T_1

    z = tf.reshape(z_1 , [-1 , tf.shape(z)[2] , tf.shape(z)[3] , tf.shape(z)[4]])  



    S_2 = Dense_Spatial_Block(z , "DSB_2")
    S_2 = Spatial_Channel_Aware_Block(S_2 , "SCA_2")    
    z_2 = S_2
        
    z = tf.reshape(z_2 , [batch_SIZE , time_step , tf.shape(z)[1] , tf.shape(z)[2] , tf.shape(z)[3] ],name="Reshape_S_T")
    
    T_2 = Dense_Temporal_Block(z , "DTB_2")
    T_2 = Temporal_Channel_Aware_Block(T_2 , "TCA_2")    
    z_2 = T_2

    z = tf.reshape(z_2 , [-1 , tf.shape(z)[2] , tf.shape(z)[3] , tf.shape(z)[4]])


    
    S_3 = Dense_Spatial_Block(z , "DSB_3")
    S_3 = Spatial_Channel_Aware_Block(S_3 , "SCA_3")    
    z_3 = S_3
        
    z = tf.reshape(z_3 , [batch_SIZE , time_step , tf.shape(z)[1] , tf.shape(z)[2] , tf.shape(z)[3] ],name="Reshape_S_T")
    
    T_3 = Dense_Temporal_Block(z , "DTB_3")
    T_3 = Temporal_Channel_Aware_Block(T_3 , "TCA_3")    
    z_3 = T_3

    z = tf.reshape(z_3 , [-1 , tf.shape(z)[2] , tf.shape(z)[3] , tf.shape(z)[4]])


    
    S_4 = Dense_Spatial_Block(z , "DSB_4")
    S_4 = Spatial_Channel_Aware_Block(S_4 , "SCA_4")    
    z_4 = S_4
        
    z = tf.reshape(z_4 , [batch_SIZE , time_step , tf.shape(z)[1] , tf.shape(z)[2] , tf.shape(z)[3] ],name="Reshape_S_T")
    
    T_4 = Dense_Temporal_Block(z , "DTB_4")
    T_4 = Temporal_Channel_Aware_Block(T_4 , "TCA_4")    
    z_4 = T_4

    z = tf.reshape(z_4 , [-1 , tf.shape(z)[2] , tf.shape(z)[3] , tf.shape(z)[4]])
    

    
        
    z = dilated_conv2d(z , 512 , 128 , 1 , "128")
    z = dilated_conv2d(z , 128 , 64  , 1 , "64")
    z = output_adjust(z , [158,238], 1 , "End")
    z = tf.reshape(z , [batch_SIZE , time_step , 158 , 238 , 1])


    all_variable = tf.trainable_variables()
    Reg_Loss = 1e-4 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in all_variable ])


    cost = tools.PRL(z , y , "L1")
    cost = cost + Reg_Loss
    
    performance = tools.compute_MAE_and_MSE(z , y)
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LR).minimize(cost)


   
    initial = tf.compat.v1.global_variables_initializer()


    with tf.compat.v1.Session() as sess:

        print("-----------------------------------------------------------------------------\n")
        print("\nStart Training...\n")
        print("Number of parameters : " , np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) , "\n")

        
        Time = 0
        seed = 0
      
        
        train_writer = tf.compat.v1.summary.FileWriter("./logs/train",sess.graph)
        test_writer  = tf.compat.v1.summary.FileWriter("./logs/test" ,sess.graph)
    
        sess.graph.finalize()    
        sess.run(initial)


        for epoch in range(Epoch+1):
            
            if epoch == 30:
                lr = lr / 2
            if epoch == 60:
                lr = lr / 2
            if epoch == 100:
                lr = lr / 2
                
            start_time = time.time()
        
            mini_batch_cost = 0
            mini_batch_MAE = 0
            mini_batch_MSE = 0
        
            seed = seed + 1
            
            minibatches = tools.random_mini_batches(X_train , Y_train  , batch_SIZE , seed=seed)
            
            for data in minibatches:
                
                (X_train_batch , Y_train_batch) = data

                _ , temp_cost ,train_performance = sess.run([optimizer , cost, performance],feed_dict={x:X_train_batch , 
                                                                                                       y:Y_train_batch ,
                                                                                                       LR:lr,
                                                                                                       })
                
    
                mini_batch_cost += temp_cost * batch_SIZE * time_step / (X_train.shape[0] * X_train.shape[1])
                mini_batch_MAE  += train_performance[0] / (X_train.shape[0] * X_train.shape[1])
                mini_batch_MSE  += train_performance[1] / (X_train.shape[0] * X_train.shape[1])
            
            
            total_cost = round(mini_batch_cost , 7)
            total_MAE  = round(mini_batch_MAE , 4)
            total_MSE  = round(np.sqrt(mini_batch_MSE) , 4)
            
            print("Epoch : ",epoch , " , Cost :  " , total_cost , " , MAE : ",  total_MAE , ", MSE : " , total_MSE)



            if True:
             
                test_cost  , test_MAE , test_MSE = 0,0,0
                test_batches = tools.random_mini_batches(X_test , Y_test , batch_SIZE , seed=seed)
    
                for i in test_batches:
                    
                    (X_test_batch , Y_test_batch ) = i
        
                    temp_cost , test_performance = sess.run([cost , performance] , feed_dict={ x:X_test_batch , 
                                                                                               y:Y_test_batch , 
                                                                                               })

                    test_cost += temp_cost * batch_SIZE * time_step / (X_test.shape[0] * X_test.shape[1])
                    test_MAE += test_performance[0] / (X_test.shape[0] * X_test.shape[1])
                    test_MSE += test_performance[1] / (X_test.shape[0] * X_test.shape[1])
        
        
                test_cost = round(test_cost , 7)
                test_MAE = round(test_MAE , 4)
                test_MSE = round(np.sqrt(test_MSE) , 4)
            
                print("Testing , cost :  " , test_cost , " , MAE : ", test_MAE," , MSE : " , test_MSE , "\n")



            process_time = time.time() - start_time        
            Time = Time + (process_time - Time) / (epoch + 1)

            if epoch % 5 == 0 :
            
                print("Average training time  per epoch : " , Time)



    print("Done.\n")


if __name__ == "__main__" :


    tf.compat.v1.reset_default_graph()

    batch_SIZE = 1
    
    time_step = 10

    Epoch = 120

    lr = 1e-4
    
    Training(batch_SIZE = batch_SIZE,
             time_step = time_step,
             Epoch = Epoch,
             lr = lr)
    
