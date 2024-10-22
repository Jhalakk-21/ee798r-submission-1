# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 23:05:10 2018
The MHF-net
@author: XieQi
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
class MyConvALayer(Layer):
    def __init__(self, in_filters, out_filters, strides, iniA, **kwargs):
        super(MyConvALayer, self).__init__(**kwargs)
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.strides = strides
        self.iniA = iniA

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='A',
            shape=[1, 1, self.in_filters, self.out_filters],
            # initializer=tf.constant_initializer(self.iniA),
            initializer=tf.keras.initializers.Constant(self.iniA),
            trainable=True,
        )

    def call(self, x):
        return tf.nn.conv2d(x, self.kernel, strides=self.strides, padding='SAME')
class MyConvALayer2(Layer):
    def __init__(self, filter, **kwargs):
        super(MyConvALayer2, self).__init__(**kwargs)
        self.filter = filter

    def call(self, G):
        return tf.nn.conv2d(G, self.filter, [1,1,1,1], padding='SAME')
# main MHF-net net
def HSInet(Y,Z, iniUp3x3,iniA,upRank,outDim,HSInetL,subnetL,ratio=32):
    

    # B = tf.get_variable(
    #           'B', 
    #           [1, 1, upRank, outDim],
    #           tf.float32, 
    #           initializer=tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1))
    B = tf.Variable(
        initial_value=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)(shape=[1, 1, upRank, outDim]),
        trainable=True,
        name='B'
    )


    tranB = tf.transpose(B,perm = [0,1,3,2])
    CListX = getCs( ratio)#inital the kernel for downsampling
    downY4, downY16, _ = downSam(Y, CListX, 3,ratio)# getPrior for upsample
    
    # fist stage
    YA = MyconvA( Y, 3, outDim, [1,1,1,1], iniA) #caculating YA
    _, _, downX32 = downSam(YA, CListX, outDim, ratio)  # downsampling 
    E  = downX32-Z 
    G  = UpSam(E, downY4, downY16, Y, iniUp3x3, outDim, ratio) # unsampling E
    G = MyConvALayer2(filter=tranB)(G)
    # G  = tf.nn.conv2d(G, tranB, [1,1,1,1], padding='SAME')
    HY = -G #
    HY  = resCNNnet(HY,1,upRank, subnetL) 
#    HY  = resCNNnetPrior(('Pri%s'%(1)),HY, Y,1,upRank,3,subnetL)
    ListX = []
    
    # 2nd to the 19th stage
    for j in range(HSInetL-2):
        # HYB= tf.nn.conv2d(HY, B, [1,1,1,1], padding='SAME')
        HYB = MyConvALayer2(filter=B)(HY)

        ListX.append(YA + HYB)
        _, _, downX32  = downSam( ListX[int(j)],CListX,outDim,  ratio)
        E   = downX32-Z
        G   = UpSam( E, downY4, downY16, Y, iniUp3x3, outDim, ratio)
        # G   = tf.nn.conv2d(G, tranB, [1,1,1,1], padding='SAME')
        G = MyConvALayer2(filter=tranB)(G)

        HY  = HY-G
        HY  = resCNNnet(HY,j+2,upRank, subnetL)
#        HY  = resCNNnetPrior(('Pri%s'%(j+2)),HY, Y,j+2,upRank,3,subnetL)
    
    #the final stage
    # HYB     = tf.nn.conv2d(HY, B, [1,1,1,1], padding='SAME')
    HYB = MyConvALayer2(filter=B)(HY)

    ListX.append(YA + HYB)
    outX    = resCNNnet(ListX[int(HSInetL-2)],101,outDim, levelN = 5)
    _,_,CX  = downSam( ListX[int(HSInetL-2)],CListX, outDim,  ratio)
    E  = CX-Z
    return outX, ListX, YA, E, HY
    
# reCNNnet 
def resCNNnet(X,j,channel,levelN):
    # with tf.variable_scope(name): 
    #     for i in range(levelN-1):
    #         X = resLevel(('resCNN_%s_%s'%(j,i+1)), 3, X, channel)                        
    #     return X    
    for i in range(levelN - 1):
        X = resLevel( 3, X, channel)
    return X

    
# reCNNnet 
def resCNNnetPrior(X,Y,j,channel,channelY,levelN):
#     with tf.variable_scope(name): 
#         for i in range(levelN-1):
# #            X = resLevel(('resCNN_%s_%s'%(j,i+1)), 3, X, channel)        
#             X = resLevel_addF(('resCNN_%s_%s'%(j,i+1)), 3, X, Y,channel,channelY)               
#         return X  
    for i in range(levelN - 1):
        X = resLevel_addF(3, X,Y, channel, channelY)

    return X
                        
# get the downsampling kernels  
def getCs(ratio):
    Clist = []
    # with tf.variable_scope(name):
    # filter1 = tf.get_variable(
    #         'Cfilter', 
    #         [6, 6, 1, 1],
    #         tf.float32, 
    #         initializer=tf.constant_initializer(1/36))
    # filter1 = tf.keras.initializers.Constant(value=1 / 36)(shape=[6, 6, 1, 1])
    filter1 = tf.Variable(
        initial_value=tf.keras.initializers.Constant(value =1 / 36) (shape=[6, 6, 1, 1]),
        trainable=True
        # name='Cfilter'
)
    Clist.append(filter1)
    if ratio >4:
        # filter2 = tf.get_variable(
        #         'Cfilter2', 
        #         [6, 6, 1, 1],
        #         tf.float32, 
        #         initializer=tf.constant_initializer(1/36)) 
        filter2 = tf.Variable(
            initial_value=tf.keras.initializers.Constant(value=1 / 36)(shape=[6, 6, 1, 1]),
            trainable=True
            # name='Cfilter2'
        )    
        Clist.append(filter2)
        if ratio>16:
                # filter3 = tf.get_variable(
                # 'Cfilter3', 
                # [3, 3, 1, 1],
                # tf.float32, 
                # initializer=tf.constant_initializer(1/9))  
            filter3 = tf.Variable(
                # initial_value=tf.keras.initializers.Constant(1 / 9, shape=[3, 3, 1, 1], dtype=tf.float32),
                initial_value= tf.keras.initializers.Constant(value=1 / 9)(shape=[3, 3, 1, 1]),
                trainable=True
                # name='Cfilter3'
            )   
            Clist.append(filter3)
    return Clist    

class DepthwiseConv2DLayer1(Layer):
    def __init__(self, Clist, k, ChDim):
        super(DepthwiseConv2DLayer1, self).__init__()
        self.Clist = Clist
        self.k = k
        self.ChDim = ChDim

    def call(self, X):
        # Use depthwise convolution within the Keras layer
        return tf.nn.depthwise_conv2d(X, 
                                      tf.tile(self.Clist[int(self.k)], [1, 1, self.ChDim, 1]), 
                                      strides=[1, 1, 1, 1], 
                                      padding='SAME')
class DepthwiseConv2DLayer2(Layer):
    def __init__(self, filter1, **kwargs):
        super(DepthwiseConv2DLayer2, self).__init__(**kwargs)
        self.filter1 = tf.convert_to_tensor(filter1, dtype=tf.float32)  # Ensure filter is a Tensor

    def call(self, X):
        return tf.nn.depthwise_conv2d(X, self.filter1, strides=[1, 1, 1, 1], padding='SAME')

# Usage inside your model
# Assuming Clist, k, and ChDim are defined elsewhere in your code

def downSam(X, Clist, ChDim, ratio):
    k=-1
    k      = k+1
    # X      = tf.nn.depthwise_conv2d(X, tf.tile(Clist[int(k)],[1,1,ChDim,1]), strides=[1,1,1,1],padding='SAME')
    filter = tf.tile(Clist[int(k)],[1,1,ChDim,1])
    X = DepthwiseConv2DLayer1(Clist=Clist, k=k, ChDim= ChDim)(X)

    downX4 = X[:,1:-1:4,1:-1:4,:]
    if ratio ==4:
        downX16 = []
        downX32 = downX4
    else: 
        k       = k+1
        filter = tf.tile(Clist[int(k)],[1,1,ChDim,1])
        X = DepthwiseConv2DLayer1(Clist=Clist, k=k, ChDim= ChDim)(downX4)
        # X       = tf.nn.depthwise_conv2d(downX4, tf.tile(Clist[int(k)],[1,1,ChDim,1]), strides=[1,1,1,1],padding='SAME')                
        downX16 = X[:,1:-1:4,1:-1:4,:]   
        if ratio==16:
            downX32 = downX16
        else:
            k  = k+1
            filter = tf.tile(Clist[int(k)],[1,1,ChDim,1])
            X = DepthwiseConv2DLayer1(Clist=Clist, k=k, ChDim= ChDim)(downX16)

            # X       = tf.nn.depthwise_conv2d(downX16, tf.tile(Clist[int(k)],[1,1,ChDim,1]), strides=[1,1,1,1],padding='SAME')         
            downX32 = X[:,0:-1:2,0:-1:2,:]      

    return downX4,  downX16,  downX32
# class DepthwiseConvLayer2(tf.keras.layers.Layer):
#     def __init__(self, filter1, **kwargs):
#         super(DepthwiseConvLayer, self).__init__(**kwargs)
#         self.filter1 = filter1

#     def call(self, X):
#         return tf.nn.depthwise_conv2d(X, self.filter1, strides=[1, 1, 1, 1], padding='SAME')

# Instantiate and use the custom layer
      
  
def UpSam(X, downY4, downY16, Y, iniUp3x3, outDim, ratio):
    if ratio==32:
        X = UpsumLevel2(X,iniUp3x3, outDim)# 2 timse upsampling
        X = resLevel_addF( 3, X, downY16/10, outDim,3)# adjusting after upsampling     

    if ratio>=16:
        X = UpsumLevel2(X,iniUp3x3, outDim)# 
        X = UpsumLevel2(X,iniUp3x3, outDim)#    
        X = resLevel_addF(3, X, downY4/10, outDim,3)#
                    
    X = UpsumLevel2(X,iniUp3x3, outDim)# 
    X = UpsumLevel2(X,iniUp3x3, outDim)# 
    X = resLevel_addF( 3, X, Y/10, outDim,3)# 
    # filter1 = tf.get_variable(
    #     'Blur', [4, 4, outDim, 1], tf.float32, initializer=tf.constant_initializer(1/16))
    # X = DepthwiseConv2DLayer(Clist, k, ChDim)(X)
    # TensorFlow 2.x version
    filter1 = tf.Variable(
        # initial_value=tf.keras.initializers.Constant(1/16, shape=[4, 4, outDim, 1], dtype=tf.float32)
        initial_value=  tf.keras.initializers.Constant(value=1 / 16)(shape=[4, 4, outDim, 1]),
        # name='Blur'
    )

    
    # X = tf.nn.depthwise_conv2d(X,filter1,strides=[1,1,1,1],padding='SAME')        
    X = DepthwiseConv2DLayer2(filter1= filter1)(X)
    return X  
        
class resLevel_extra(Layer):
    def __init__(self, kernel, strides,padding = 'SAME', Channel=None, Fsize=None,**kwargs):
        super(resLevel_extra, self).__init__(**kwargs)
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.Channel = Channel
        self.Fsize = Fsize
        
        self.kernel3 = self.add_weight(
            # name = 'weights3',
            shape = [self.Fsize, self.Fsize, self.Channel+3, self.Channel+3],
            # initializer='truncated_normal',
            initializer=tf.keras.initializers.HeNormal(),

            trainable=True
        )
        self.biases3 = self.add_weight(
            # name = 'biases3',
            shape = [self.Channel+3],
            # initializer='zeros',
            initializer=tf.keras.initializers.Zeros(),  

            trainable=True
        )
        self.scale3 = self.add_weight(
            # name = 'scale3',
            shape = [self.Channel+3],
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )
        self.beta3 = self.add_weight(
            # name = 'beta3',
            shape = [self.Channel+3],
            initializer=tf.keras.initializers.Zeros(),  
            trainable=True
        )
        self.kernel2=self.add_weight(
            # name = 'weights2',
            shape=[self.Fsize, self.Fsize, self.Channel+3, self.Channel],
            trainable = True,
            initializer=tf.keras.initializers.HeNormal(),
        )
        self.biases2=self.add_weight(
            # name='biases2',
            shape = [self.Channel],
            # initializer='zeros',
            initializer=tf.keras.initializers.Zeros(),  

            trainable= True

        )
        self.scale2 = self.add_weight(
            # name = 'scale2',
            shape = [self.Channel],
            initializer=tf.keras.initializers.Ones(),
            trainable= True
        )
        self.beta2 = self.add_weight(
            # name = 'beta2',
            shape = [self.Channel],
            # initializer='zeros'
            initializer=tf.keras.initializers.Zeros(), 

            trainable=True

        )

    def call(self, X, biases, beta, scale):
        conv = tf.nn.conv2d(X, self.kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 加到新的test里
        # kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, Channel+3])
        # biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases3')
        # scale = tf.Variable(tf.ones([Channel+3])/20, trainable=True, name=('scale3'))
        # beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta3'))

        conv3 = tf.nn.conv2d(feature_relu, self.kernel3, [1, 1, 1, 1], padding='SAME')
        feature3 = tf.nn.bias_add(conv3, self.biases3)
        
        mean3, var3  = tf.nn.moments(feature3,[0, 1, 2])
        feature_normal3 = tf.nn.batch_normalization(feature3, mean3, var3, self.beta3, self.scale3/20, 1e-5)

        feature_relu3 = tf.nn.relu(feature_normal3)
        
        # 加到新的test里
        
        
        # kernel3 = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel])
        # biases = tf.Variable(tf.constant(0.0, shape=[Channel], dtype=tf.float32), trainable=True, name='biases2')
        # scale = tf.Variable(tf.ones([Channel])/20, trainable=True, name=('scale2'))
        # beta = tf.Variable(tf.zeros([Channel]), trainable=True, name=('beta2'))

        conv2 = tf.nn.conv2d(feature_relu3, self.kernel2, [1, 1, 1, 1], padding='SAME')
        feature2= tf.nn.bias_add(conv2, self.biases2)

        mean2, var2  = tf.nn.moments(feature2,[0, 1, 2])
        feature_normal2 = tf.nn.batch_normalization(feature2, mean2, var2, self.beta2, self.scale2/20, 1e-5)

        feature_relu2 = tf.nn.relu(feature_normal2)        


        X = tf.add(X, feature_relu2)  #  shortcut  
        return X
def resLevel(Fsize, X, Channel):
    # 两层调整
    kernel = create_kernel( shape=[Fsize, Fsize, Channel, Channel+3])
    # biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True)
    # scale = tf.Variable(tf.ones([Channel+3])/20, trainable=True)
    # beta = tf.Variable(tf.zeros([Channel+3]), trainable=True)
    # kernel = tf.Variable(
    #     tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)(shape=[Fsize, Fsize, Channel, Channel+3]),
    #     trainable=True
    # )
    biases = tf.Variable(
        tf.keras.initializers.Zeros()(shape=[Channel+3]),
        trainable=True
    )
    scale = tf.Variable(
        tf.keras.initializers.Ones()(shape=[Channel+3]) / 20,
        trainable=True
    )
    beta = tf.Variable(
        tf.keras.initializers.Zeros()(shape=[Channel+3]),
        trainable=True
)

    func1 = resLevel_extra( kernel=kernel, strides = [1,1,1,1],padding='SAME', Channel=Channel, Fsize=Fsize)

    conv1 = func1(X, biases = biases, beta=beta, scale = scale)
    return conv1
    # conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
    # feature = tf.nn.bias_add(conv, biases)

    # mean, var  = tf.nn.moments(feature,[0, 1, 2])
    # feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

    # feature_relu = tf.nn.relu(feature_normal)
    
    # # 加到新的test里
    # kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, Channel+3])
    # biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases3')
    # scale = tf.Variable(tf.ones([Channel+3])/20, trainable=True, name=('scale3'))
    # beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta3'))

    # conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
    # feature = tf.nn.bias_add(conv, biases)
    
    # mean, var  = tf.nn.moments(feature,[0, 1, 2])
    # feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

    # feature_relu = tf.nn.relu(feature_normal)
    
    # # 加到新的test里
    
    
    # kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel])
    # biases = tf.Variable(tf.constant(0.0, shape=[Channel], dtype=tf.float32), trainable=True, name='biases2')
    # scale = tf.Variable(tf.ones([Channel])/20, trainable=True, name=('scale2'))
    # beta = tf.Variable(tf.zeros([Channel]), trainable=True, name=('beta2'))

    # conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
    # feature = tf.nn.bias_add(conv, biases)

    # mean, var  = tf.nn.moments(feature,[0, 1, 2])
    # feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

    # feature_relu = tf.nn.relu(feature_normal)        


    # X = tf.add(X, feature_relu)  #  shortcut  
    return X
class reslevel_addf_extra(Layer):
    def __init__(self, kernel, strides, padding='SAME', Channel=None, Fsize=None, ChannelX=None,**kwargs):
        super(reslevel_addf_extra, self).__init__(**kwargs)
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.Channel = Channel
        self.Fsize = Fsize
        self.ChannelX = ChannelX

        # Initialize kernel, biases, scale, and beta during __init__
        self.kernel2 = self.add_weight(
            # name='weights2',
            shape=[self.Fsize, self.Fsize, self.Channel+3, self.Channel+3],
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True
        )
        self.biases2 = self.add_weight(
            # name='biases2',
            shape=[self.Channel+3],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        self.scale2 = self.add_weight(
            # name='scale2',
            shape=[self.Channel+3],
            initializer= tf.keras.initializers.Ones(),
            trainable=True
        )
        self.beta2 = self.add_weight(
            # name='beta2',
            shape=[self.Channel+3],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        self.kernel3 = self.add_weight(
            # name = 'weights3',
            shape = [self.Fsize, self.Fsize, self.Channel+3, self.ChannelX],
            trainable = True,
            initializer=tf.keras.initializers.HeNormal(),

        )
        self.biases3 = self.add_weight(
            # name = 'baises2',
            shape = [self.ChannelX],
            initializer=tf.keras.initializers.Zeros(),
            trainable = True
        )
        self.scale3 = self.add_weight(
            # name = 'scale3',
            shape = [self.ChannelX],
            initializer=tf.keras.initializers.Ones(),
            trainable= True
        )
        self.beta3 = self.add_weight(
            # name = 'beta3',
            shape = [self.ChannelX],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True

        )

    def call(self, inputs, biases, beta, scale, Channel, Fsize):
        # inputs is a list where inputs[0] = X, inputs[1] = Y
        concatenated = tf.concat(inputs, axis=3)
        # print(f" concatenated ka shape {concatenated.shape}")
        conv = tf.nn.conv2d(concatenated, self.kernel, strides=self.strides, padding=self.padding)
        # print(f" conv ka shape {conv.shape}")

        feature = tf.nn.bias_add(conv, biases)
        mean, var = tf.nn.moments(feature, [0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
        feature_relu = tf.nn.relu(feature_normal)

        # Use the pre-initialized kernel2, biases2, scale2, and beta2
        conv2 = tf.nn.conv2d(feature_relu, self.kernel2, [1, 1, 1, 1], padding='SAME')
        feature2 = tf.nn.bias_add(conv2, self.biases2)
        mean2, var2 = tf.nn.moments(feature2, [0, 1, 2])
        feature_normal2 = tf.nn.batch_normalization(feature2, mean2, var2, self.beta2, self.scale2/100, 1e-5)

        feature_relu2 = tf.nn.relu(feature_normal2)
        # print(f"feature relu2 ka shape {feature_relu2.shape}")

        # Assuming X is the first input (inputs[0])
        # X = tf.add(inputs[0], feature_relu2)  # shortcut
        # kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, ChannelX])
        # biases = tf.Variable(tf.constant(0.0, shape=[ChannelX], dtype=tf.float32), trainable=True, name='biases3')
        # scale = tf.Variable(tf.ones([ChannelX])/100, trainable=True, name=('scale3'))
        # beta = tf.Variable(tf.zeros([ChannelX]), trainable=True, name=('beta3'))

        conv = tf.nn.conv2d(feature_relu2, self.kernel3, [1, 1, 1, 1], padding = 'SAME')
        feature3 = tf.nn.bias_add(conv, self.biases3)

        mean3, var3  = tf.nn.moments(feature3,[0, 1, 2])
        feature_normal3 = tf.nn.batch_normalization(feature3, mean3, var3, self.beta3, self.scale3/100, 1e-5)

        feature_relu3 = tf.nn.relu(feature_normal3)
        # print(f"feature relu3 ka shape {feature_relu3.shape}")
        X = tf.add(inputs[0], feature_relu3)  #  shortcut  

        return X

def resLevel_addF(Fsize, X, Y, ChannelX, ChannelY):
    Channel = ChannelX + ChannelY
    kernel = create_kernel( shape=[Fsize, Fsize, Channel, Channel+3])
    # biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True)
    # scale = tf.Variable(tf.ones([Channel+3]) / 100, trainable=True)
    # beta = tf.Variable(tf.zeros([Channel+3]), trainable=True)
    # kernel = tf.Variable(
    #     tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)(shape=[Fsize, Fsize, Channel, Channel+3]),
    #     trainable=True
    # )
    biases = tf.Variable(
        tf.keras.initializers.Zeros()(shape=[Channel+3]),
        trainable=True
    )
    scale = tf.Variable(
        tf.keras.initializers.Ones()(shape=[Channel+3]) / 100,
        trainable=True
    )
    beta = tf.Variable(
        tf.keras.initializers.Zeros()(shape=[Channel+3]),
        trainable=True
    )

    # Create an instance of the custom layer with pre-initialized weights
    func = reslevel_addf_extra(kernel=kernel, strides=[1, 1, 1, 1], Channel=Channel, Fsize=Fsize, ChannelX= ChannelX)
    # print(X.shape, Y.shape)
    # Call the conv2d layer with inputs X and Y, passing additional arguments as keyword arguments
    conv = func([X, Y], biases=biases, beta=beta, scale=scale, Channel=Channel, Fsize=Fsize)

    return conv

# class reslevel_addf_extra(Layer):
#     def __init__(self, kernel, strides, padding='SAME', **kwargs):
#         super(reslevel_addf_extra, self).__init__(**kwargs)
#         self.kernel = kernel
#         self.strides = strides
#         self.padding = padding

#     def call(self, inputs, biases, beta, scale, Channel, Fsize):
#         # inputs is a list where inputs[0] = X, inputs[1] = Y
#         concatenated = tf.concat(inputs, axis=3)
#         conv = tf.nn.conv2d(concatenated, self.kernel, strides=self.strides, padding=self.padding)
#         feature = tf.nn.bias_add(conv, biases)
#         mean,var = tf.nn.moments(feature, [0,1,2])
#         feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
#         feature_relu = tf.nn.relu(feature_normal)

#         kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel+3])
#         biases = tf.Variable(tf.constant(0.0, shape=[ Channel+3], dtype=tf.float32), trainable=True, name='biases2')
#         scale = tf.Variable(tf.ones([ Channel+3])/100, trainable=True, name=('scale2'))
#         beta = tf.Variable(tf.zeros([ Channel+3]), trainable=True, name=('beta2'))

#         conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
#         feature = tf.nn.bias_add(conv, biases)

#         mean, var  = tf.nn.moments(feature,[0, 1, 2])
#         feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

#         feature_relu = tf.nn.relu(feature_normal)

#         X = tf.add(X, feature_relu)  #  shortcut  

#         return X

# def resLevel_addF(name, Fsize, X, Y,ChannelX,ChannelY):
#     # with tf.variable_scope(name):
#         # 两层调整
#     Channel = ChannelX+ChannelY
#     kernel = create_kernel(name='weights1', shape=[Fsize, Fsize, Channel, Channel+3])
#     biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases1')
#     scale = tf.Variable(tf.ones([Channel+3])/100, trainable=True, name=('scale1'))
#     beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta1'))

#     # conv = tf.nn.conv2d(tf.concat([X,Y],3), kernel, [1, 1, 1, 1], padding='SAME')
#     func = reslevel_addf_extra(kernel=kernel, strides=[1, 1, 1, 1])

#     # Call the conv2d layer with inputs X and Y
#     conv = func([X, Y], biases=biases, beta=beta, scale=scale, Channel=Channel, Fsize=Fsize)

#     # feature = tf.nn.bias_add(conv, biases)

#     # mean, var  = tf.nn.moments(feature,[0, 1, 2])
#     # feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

#     # feature_relu = tf.nn.relu(feature_normal)
    
#     # # 我又加了一层
#     # kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel+3])
#     # biases = tf.Variable(tf.constant(0.0, shape=[ Channel+3], dtype=tf.float32), trainable=True, name='biases2')
#     # scale = tf.Variable(tf.ones([ Channel+3])/100, trainable=True, name=('scale2'))
#     # beta = tf.Variable(tf.zeros([ Channel+3]), trainable=True, name=('beta2'))

#     # conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
#     # feature = tf.nn.bias_add(conv, biases)

#     # mean, var  = tf.nn.moments(feature,[0, 1, 2])
#     # feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
#     # feature_relu = tf.nn.relu(feature_normal)
#     # #
    
#     # kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, ChannelX])
#     # biases = tf.Variable(tf.constant(0.0, shape=[ChannelX], dtype=tf.float32), trainable=True, name='biases3')
#     # scale = tf.Variable(tf.ones([ChannelX])/100, trainable=True, name=('scale3'))
#     # beta = tf.Variable(tf.zeros([ChannelX]), trainable=True, name=('beta3'))

#     # conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
#     # feature = tf.nn.bias_add(conv, biases)

#     # mean, var  = tf.nn.moments(feature,[0, 1, 2])
#     # feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

#     # feature_relu = tf.nn.relu(feature_normal)

#     # X = tf.add(X, feature_relu)  #  shortcut  
#     return conv    
    
def ConLevel(name, Fsize, X, inC, outC):
    # with tf.variable_scope(name):
    kernel = create_kernel( shape=[Fsize, Fsize, inC, outC])
    # biases = tf.Variable(tf.constant(0.0, shape=[outC], dtype=tf.float32), trainable=True)
    # scale = tf.Variable(tf.ones([outC]), trainable=True)
    # beta = tf.Variable(tf.zeros([outC]), trainable=True)
    # kernel = tf.Variable(
    #     tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)(shape=[Fsize, Fsize, Channel, Channel+3]),
    #     trainable=True
    # )
    biases = tf.Variable(
        tf.keras.initializers.Zeros()(shape=[outC]),
        trainable=True
    )
    scale = tf.Variable(
        tf.keras.initializers.Ones()(shape=[outC]),
        trainable=True
    )
    beta = tf.Variable(
        tf.keras.initializers.Zeros()(shape=[outC]),
        trainable=True
    )

    conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
    X = tf.nn.bias_add(conv, biases)

    mean, var = tf.nn.moments(X,[0, 1, 2])
    X = tf.nn.batch_normalization(X,  mean, var, beta, scale, 1e-5)

    X = tf.nn.relu(X)
    return X
# import tensorflow as tf
# from tensorflow.keras.layers import Layer

# class mimport tensorflow as tf
# from tensorflow.keras.layers import Layer

class UpsampleLayer(Layer):
    def __init__(self, iniUp2x2, outDim):
        super(UpsampleLayer, self).__init__()
        self.outDim = outDim
        # Initialize the filter for the upsampling
        # self.filter1 = tf.Variable(
        #     initial_value=tf.constant(iniUp2x2 / 4, shape=[3, 3, outDim, outDim], dtype=tf.float32),
        #     trainable=True
        #     # name=name
        # )
        self.filter1 = tf.Variable(
            initial_value=tf.keras.initializers.Constant(value=iniUp2x2 / 4)(shape=[3, 3, outDim, outDim]),
            trainable=True
        )


    def call(self, X):
        # Get the input shape dynamically within the call method
        input_shape = tf.shape(X)
        
        # Calculate the output shape by scaling the height and width by 2
        output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, self.outDim]
        
        # Perform the transpose convolution (upsampling) operation
        X = tf.nn.conv2d_transpose(X, self.filter1, output_shape, strides=[1, 2, 2, 1], padding='SAME')
        return X

# Now use this layer within your code
def UpsumLevel2(X, iniUp2x2, outDim):
    # Create an instance of the custom upsample layer
    upsample_layer = UpsampleLayer(iniUp2x2, outDim)
    
    # Call the layer with input X
    return upsample_layer(X)



# class MyConvALayer(Layer):
#     def __init__(self, in_filters, out_filters, strides, iniA, **kwargs):
#         super(MyConvALayer, self).__init__(**kwargs)
#         self.in_filters = in_filters
#         self.out_filters = out_filters
#         self.strides = strides
#         self.iniA = iniA

#     def build(self, input_shape):
#         self.kernel = self.add_weight(
#             name='A',
#             shape=[1, 1, self.in_filters, self.out_filters],
#             initializer=tf.constant_initializer(self.iniA),
#             trainable=True,
#         )

#     def call(self, x):
#         return tf.nn.conv2d(x, self.kernel, strides=self.strides, padding='SAME')

# Usage
def MyconvA(x, in_filters, out_filters, strides, iniA):
    return MyConvALayer(in_filters, out_filters, strides, iniA)(x)

# A 1X1 convolution for caculating Y*A
# def MyconvA( name, x, in_filters, out_filters, strides, iniA):
#     # with tf.variable_scope(name):
         
#     kernel = tf.Variable(
#             name = 'A', 
#             initial_value=tf.constant(iniA, shape =[1, 1, in_filters, out_filters], dtype = tf.float32),
#     #              initializer=tf.constant_initializer(1))
#             trainable= True
#             )
#     return tf.nn.conv2d(x, kernel, strides, padding='SAME')
  
    
# B 1X1 convolution for caculating Y_hat*B  
def MyconvB( name, x, in_filters, out_filters, strides):
    # with tf.variable_scope(name):
    kernel = tf.Variable(
        initial_value=tf.random.truncated_normal([1, 1, in_filters, out_filters], mean=0.0, stddev=0.1),
        trainable=True,
        name='B'
    )
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')    

# def create_kernel(name, shape, initializer=tf.initializers.truncated_normal(mean = 0, stddev = 0.1)):
# #def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
#     regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)

#     new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
#                                     regularizer=regularizer, trainable=True)
#     # new_variables = tf.Variable()
#     return new_variables


def create_kernel(shape, initializer=tf.keras.initializers.HeNormal()):
    # Use L2 regularizer from tf.keras
    regularizer = tf.keras.regularizers.l2(l2=1e-10)

    # Create the kernel variable with the initializer and regularizer
    new_variables = tf.Variable(
        initial_value=initializer(shape=shape),
        trainable=True,
        regularizer=regularizer
    )
    return new_variables
