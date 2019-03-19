# -*- coding:  UTF-8  -*-
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import os
import Param

##########################
# c 是指定路径下的子目录,在此处也是标签名
# a 是字典,字典的关键字是子目录,键值是数字(以0为始,逐次加1)
# i 是数字,以0为始,逐次加1,此处是指代类别
# l 是具体文件夹下,具体每一张图片名
# path
# |_c
#   |_l
# 每一张图片的路径:image_path = path+'/'+c+'/'+l
##########################
def shuffle_train_file(path):
    Image_list = []
    a = {}
    i = 0
    for c in os.listdir(path):
        a.update(c=i)
        if (i<2):
           for l in os.listdir(path+'/'+c):
               list_tmp=[]
               image_path = path+'/'+c+'/'+l
               list_tmp.append(image_path)
               list_tmp.append(str(a['c']))
               Image_list.append(list_tmp)
        i = i + 1

    Image_Array = np.array(Image_list,dtype='str')
    np.random.shuffle(Image_Array)

    return Image_Array
############################
# 创建train.tfrecords文件
# 创建fileQueue_train文件,文件中存放用于训练所有图片(以换行符隔开)的路径
# 返回值:length是文件的行数,同时也是进行训练的图片数
############################
def Create_Train_TFRecord():
    train_image_path='/bigdata/tenglong/VGG16/train'
    filename=('/home/htl/VGG16_htl/Imagenet_tfrecord/train.tfrecords')
    writer=tf.python_io.TFRecordWriter(filename)
    print('start to convert train_dataset')
    Image_Array=shuffle_train_file(train_image_path)
    length=0
    with open('/home/htl/VGG16_htl/Imagenet_tfrecord/fileQueue_train','w') as f:
        for img in Image_Array:
            index=img[1]
            img_raw=Image.open(img[0])

            if(img_raw.mode!='RGB'):
                print('Not RGB',img[0])
                continue

            if((img_raw.size[0]<64)or(img_raw.size[1]<64)):
                print('Size is too small',img[0])
                continue

            img_raw=img_raw.resize((Param.Height,Param.Width))
            img_raw_new=img_raw.tobytes()

            f.writelines(img[0]+'\n')
            length=length+1

            example=tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
                            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_new]))
                            }
                        )
                    )
            writer.write(example.SerializeToString())
        writer.close()
    print('train_dataset convert end')

    return length
############################
# 创建test.tfecords文件
# 创建fileQueue_test文件,文件中存放用于测试所有图片(以换行符隔开)的路径
# 返回值:length是文件的行数,同时也是进行测试的图片数
############################
def Create_Val_TFRecord():
    val_image_path='/bigdata/tenglong/VGG16/test'
    filename=('/home/htl/VGG16_htl/Imagenet_tfrecord/test.tfrecords')
    writer=tf.python_io.TFRecordWriter(filename)
    print('start to convert test_dataset')
    Image_Array=shuffle_train_file(val_image_path)
    length=0
    with open('/home/htl/VGG16_htl/Imagenet_tfrecord/fileQueue_test','w') as f:
        for img in Image_Array:
            index=img[1]
            img_raw=Image.open(img[0])

            if(img_raw.mode!='RGB'):
                print('Not RGB',img[0])
                continue

            if((img_raw.size[0]<64)or(img_raw.size[1]<64)):
                print('Size is too small',img[0])
                continue

            img_raw=img_raw.resize((Param.Height,Param.Width))
            img_raw_new=img_raw.tobytes()

            f.writelines(img[0]+'\n')
            length=length+1

            example=tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
                            'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_new]))
                            }
                        )
                    )
            writer.write(example.SerializeToString())
        writer.close()
    print('val_dataset convert end')

    return length
############################
#从train.tfrecords文件中读取训练用的图片及其对应的标签
############################
def Read_Train_TFRecord():
      TFRecord_file='/home/htl/VGG16_htl/Imagenet_tfrecord/train.tfrecords'
      filename_queue=tf.train.string_input_producer([TFRecord_file])
      reader=tf.TFRecordReader()
      _,example=reader.read(filename_queue)
      features=tf.parse_single_example(example,features={'label':tf.FixedLenFeature([],tf.int64),'img_raw':tf.FixedLenFeature([],tf.string)})
      images=tf.decode_raw(features['img_raw'],tf.uint8)
      images=tf.reshape(images,[Param.Height,Param.Width,Param.Channel])
      labels=tf.cast(features['label'],tf.int32)

      images,labels=tf.train.shuffle_batch([images,labels],batch_size=Param.Batch,capacity=16*Param.Batch,min_after_dequeue=4*Param.Batch,num_threads=Param.Threads_num)
      labels=tf.one_hot(labels,Param.Class_Num)

      print('Train read over')
      return images,labels
############################
#从test.tfrecords文件中读取测试用的图片及其对应的标签
############################
def Read_Val_TFRecord():
      TFRecord_file='/home/caijun/Imagenet_tfrecord/test.tfrecords'
      filename_queue=tf.train.string_input_producer([TFRecord_file])
      reader=tf.TFRecordReader()
      _,example=reader.read(filename_queue)
      features=tf.parse_single_example(example,features={'label':tf.FixedLenFeature([],tf.int64),'img_raw':tf.FixedLenFeature([],tf.string)})
      images=tf.decode_raw(features['img_raw'],tf.uint8)
      images=tf.reshape(images,[Param.Height,Param.Width,Param.Channel])
      labels=tf.cast(features['label'],tf.int32)

      images,labels=tf.train.shuffle_batch([images,labels],batch_size=Param.Batch,capacity=16*Param.Batch,min_after_dequeue=4*Param.Batch,num_threads=Param.Threads_num)
      labels=tf.one_hot(labels,Param.Class_Num)

      print('Val read over')
      return images,labels


if __name__=='__main__':
    print(Create_Train_TFRecord())
    print(Create_Val_TFRecord())
    print(Read_Train_TFRecord())
    print(Read_Val_TFRecord())
#    return_value = shuffle_train_file()
#    print (return_value)
