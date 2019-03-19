import tensorflow as tf

class CNN_Net(object):
    def __init__(self,class_num,Image_W,Image_H,image_channel):
        self._Image_Channel=image_channel
        self._Class_Num=class_num
        self._Image_Size_W=Image_W
        self._Image_Size_H=Image_H
        # self._Reshape_Size=(int(Image_W/32))*(int(Image_H/32))*512

    def _variable_weight(self,_name="weight",_shape=None,_stddev=0.02,_wd=None):
        _intinitializer=tf.truncated_normal_initializer(stddev=_stddev)
        weight=tf.get_variable(name=_name,shape=_shape,initializer=_intinitializer)

        if(_wd!=None):
            weight_decay =tf.multiply(tf.nn.l2_loss(weight), _wd, name='weight_loss')
            tf.add_to_collection('loss', weight_decay)
        return weight

    def _conv_layer(self,_name,x,Kernel_W,Kernel_H,Stride_X,Stride_Y,OutputDim):
        with tf.variable_scope(_name) as scope:
            Inputdim=x.get_shape()[-1]
            kernel=self._variable_weight(_shape=[Kernel_H,Kernel_W,Inputdim,OutputDim])
            bias=tf.get_variable(name="bias",shape=[OutputDim],initializer=tf.constant_initializer(0.0))

            conv=tf.nn.conv2d(x,kernel,strides=[1,Stride_X,Stride_Y,1],padding="SAME")
            out=tf.nn.bias_add(conv,bias)
            conv_out=tf.nn.relu(out,name=scope.name)
            return conv_out

    def _dropout_layer(self,_name,x,_keepPro):
        return tf.nn.dropout(x,_keepPro,name=_name)

    def _pool_layer(self,_name,x,Kernel_shape,Stride_shape,_padding='SAME',pooling_Mode='Max_Pool'):
        if pooling_Mode=='Max_Pool':
            return tf.nn.max_pool(x,Kernel_shape,Stride_shape,padding=_padding,name=_name)
        if pooling_Mode=='Avg_Pool':
            return tf.nn.avg_pool(x,Kernel_shape,Stride_shape,padding=_padding,name=_name)

    def _FC_layer(self,_name,x,outputDim,Stddiv,Wd):
        with tf.variable_scope(_name) as scope:
            Inputdim=x.get_shape()[-1]
            _wieght=self._variable_weight(_shape=[Inputdim,outputDim],_stddev=Stddiv,_wd=Wd)
            _b=tf.get_variable(name="bias",shape=[outputDim],initializer=tf.constant_initializer(0.01))
            fcreturn=tf.nn.relu(tf.matmul(x,_wieght)+_b,name=scope.name)
            return fcreturn
        # return tf.layers.dense(x,outputDim,activation=tf.nn.relu,name=_name)

    def _Inception_layer(self,_name,x,conv_11_size,conv_33_reduce_size,conv_33_size,conv_55_reduce_size,conv_55_size,pool_size):
        with tf.variable_scope(_name) as scope:
            conv_11=self._conv_layer((_name+'conv_11'),x,1,1,1,1,conv_11_size)

            conv_33_reduce=self._conv_layer((_name+'conv_33_reduce'),x,1,1,1,1,conv_33_reduce_size)
            conv_33=self._conv_layer((_name+'conv_33'),conv_33_reduce,3,3,1,1,conv_33_size)

            conv_55_reduce=self._conv_layer((_name+'conv_55_reduce'),x,1,1,1,1,conv_55_reduce_size)
            conv_55=self._conv_layer((_name+'conv_55'),conv_55_reduce,5,5,1,1,conv_55_size)

            pool=self._pool_layer((_name+'pool'),x,[1,3,3,1],[1,1,1,1])
            conv_pool=self._conv_layer((_name+'conv_pool'),pool,1,1,1,1,pool_size)

            return tf.concat([conv_11,conv_33,conv_55,conv_pool],3,name=scope.name)

    def Easy_CNN_Net(self):
        with tf.name_scope('data'):
            x = tf.placeholder(tf.float32, shape=[None, self._Image_Size_W,self._Image_Size_H,self._Image_Channel], name='Input')
            y = tf.placeholder(tf.float32, shape=[None, self._Class_Num], name='Output')

        cov_1_1=self._conv_layer('conv1_1',x,3,3,1,1,32)
        cov_1_2=self._conv_layer('conv1_2',cov_1_1,3,3,1,1,64)
        pool_1=self._pool_layer('pool_1',cov_1_2,[1,2,2,1],[1,2,2,1])

        cov2_1=self._conv_layer('cov2_1',pool_1,3,3,1,1,128)
        cov2_2=self._conv_layer('cov2_2',cov2_1,3,3,1,1,128)
        pool_2=self._pool_layer('pool_2',cov2_2,[1,2,2,1],[1,2,2,1])

        _Reshape_Size=(int(self._Image_Size_W/4))*(int(self._Image_Size_H/4))*128

        fc_input_image=tf.reshape(pool_2,[-1,_Reshape_Size])
        fc_1=self._FC_layer('fc_1',fc_input_image,1024,Stddiv=0.04,Wd=0.004)
        drop_1=self._dropout_layer('drop_1',fc_1,_keepPro=0.8)

        fc_2=self._FC_layer('fc_2',drop_1,512,Stddiv=0.04,Wd=0.004)
        drop_2=self._dropout_layer('drop_2',fc_2,_keepPro=0.8)

        with tf.variable_scope('softmax_layer') as scope:
            _w=self._variable_weight(_shape=[512,self._Class_Num])
            _b=tf.get_variable(name="b",shape=[self._Class_Num],initializer=tf.constant_initializer(0.0))
            # x_return=tf.nn.softmax(tf.matmul(drop_2,_w)+_b,name=scope.name)
            x_return=tf.matmul(drop_2,_w)+_b

        return x_return,x,y

    def VGG16(self):
        with tf.name_scope('data'):
            x = tf.placeholder(tf.float32, shape=[None, self._Image_Size_W,self._Image_Size_H,self._Image_Channel], name='Input')
            y = tf.placeholder(tf.float32, shape=[None, self._Class_Num], name='Output')

        cov_1_1=self._conv_layer('conv1_1',x,3,3,1,1,64)
        cov_1_2=self._conv_layer('conv1_2',cov_1_1,3,3,1,1,64)
        pool_1=self._pool_layer('pool_1',cov_1_2,[1,2,2,1],[1,2,2,1])

        cov2_1=self._conv_layer('cov2_1',pool_1,3,3,1,1,128)
        cov2_2=self._conv_layer('cov2_2',cov2_1,3,3,1,1,128)
        pool_2=self._pool_layer('pool_2',cov2_2,[1,2,2,1],[1,2,2,1])

        cov3_1=self._conv_layer('cov3_1',pool_2,3,3,1,1,256)
        cov3_2=self._conv_layer('cov3_2',cov3_1,3,3,1,1,256)
        cov3_3=self._conv_layer('cov3_3',cov3_2,3,3,1,1,256)
        pool_3=self._pool_layer('pool_3',cov3_3,[1,2,2,1],[1,2,2,1])

        cov4_1=self._conv_layer('cov4_1',pool_3,3,3,1,1,512)
        cov4_2=self._conv_layer('cov4_2',cov4_1,3,3,1,1,512)
        cov4_3=self._conv_layer('cov4_3',cov4_2,3,3,1,1,512)
        pool_4=self._pool_layer('pool_4',cov4_3,[1,2,2,1],[1,2,2,1])

        cov5_1=self._conv_layer('con5_1',pool_4,3,3,1,1,512)
        cov5_2=self._conv_layer('cov5_2',cov5_1,3,3,1,1,512)
        cov5_3=self._conv_layer('cov5_3',cov5_2,3,3,1,1,512)
        pool_5=self._pool_layer('pool_5',cov5_3,[1,2,2,1],[1,2,2,1])

        _Reshape_Size=(int(self._Image_Size_W/32))*(int(self._Image_Size_H/32))*512

        fc_input_image=tf.reshape(pool_5,[-1,_Reshape_Size])
        fc_1=self._FC_layer('fc_1',fc_input_image,4096,Stddiv=0.04,Wd=0.004)
        drop_1=self._dropout_layer('drop_1',fc_1,_keepPro=0.8)

        fc_2=self._FC_layer('fc_2',drop_1,4096,Stddiv=0.04,Wd=0.004)
        drop_2=self._dropout_layer('drop_2',fc_2,_keepPro=0.8)

        fc_3=self._FC_layer('fc_3',drop_2,self._Class_Num,Stddiv=0.04,Wd=0.004)
        return fc_3,x,y


    def GoogleNet(self):
        with tf.name_scope('data'):
            x = tf.placeholder(tf.float32, shape=[None, self._Image_Size_W,self._Image_Size_H,self._Image_Channel], name='Input')
            y = tf.placeholder(tf.float32, shape=[None, self._Class_Num], name='Output')

        conv_1=self._conv_layer('conv_1',x,7,7,2,2,64)
        max_pool_1=self._pool_layer('max_pool_1',conv_1,[1,3,3,1],[1,2,2,1])

        conv_2=self._conv_layer('conv_2',max_pool_1,3,3,1,1,192)
        conv_3=self._conv_layer('conv_3',conv_2,3,3,1,1,192)
        max_pool_2=self._pool_layer('max_pool_2',conv_3,[1,3,3,1],[1,2,2,1])

        inception_3a=self._Inception_layer('inception_3a',max_pool_2,64,96,128,16,32,32)
        inception_3b=self._Inception_layer('inception_3b',inception_3a,128,128,192,32,96,64)
        max_pool_3=self._pool_layer('max_pool_3',inception_3b,[1,3,3,1],[1,2,2,1])

        inception_4a=self._Inception_layer('inception_4a',max_pool_3,192,96,208,16,48,64)
        inception_4b=self._Inception_layer('inception_4b',inception_4a,160,112,224,24,64,64)
        inception_4c=self._Inception_layer('inception_4c',inception_4b,128,128,256,24,64,64)
        inception_4d=self._Inception_layer('inception_4d',inception_4c,112,144,288,32,64,64)
        inception_4e=self._Inception_layer('inception_4e',inception_4d,256,160,320,32,128,128)
        max_pool_4=self._pool_layer('max_pool_4',inception_4e,[1,3,3,1],[1,2,2,1])

        inception_5a=self._Inception_layer('inception_5a',max_pool_4,256,160,320,32,128,128)
        inception_5b=self._Inception_layer('inception_5b',inception_5a,384,192,384,48,128,128)

        # avg_pool_1=self._pool_layer('avg_pool_1',inception_5b,[1,7,7,1],[1,1,1,1],pooling_Mode='Avg_Pool')
        avg_pool_1=tf.nn.avg_pool(inception_5b,[1,7,7,1],[1,1,1,1],padding='VALID',name='avg_pool')

        # dropout_1=self._dropout_layer('dropout_1',avg_pool_1,_keepPro=0.4)
        # linear_1=self._conv_layer('linear-1',avg_pool_1,1,1,1,1,self._Class_Num)

        # linear_1=tf.reshape(linear_1,[-1,1*1*self._Class_Num])

        fc_input=tf.reshape(avg_pool_1,[-1,1024])

        dropout_1=self._dropout_layer('dropout_1',fc_input,_keepPro=0.5)

        with tf.variable_scope('linear_1') as scope:
            Inputdim=dropout_1.get_shape()[-1]
            _w=self._variable_weight(_shape=[Inputdim,self._Class_Num],_wd=0.004)
            _b=tf.get_variable(name="b",shape=[self._Class_Num],initializer=tf.constant_initializer(0.0))
            # x_return=tf.nn.softmax(tf.matmul(drop_2,_w)+_b,name=scope.name)
            linear_1=tf.matmul(dropout_1,_w)+_b

        # softmax_1=tf.nn.softmax(linear_1)
        return linear_1,x,y












