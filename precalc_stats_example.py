import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import warnings
import numpy as np
import fid
import tqdm
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
from cv2 import imread,resize,INTER_CUBIC,cvtColor,COLOR_BGR2RGB
import pandas as pd
import numpy.lib as npl
from PIL import Image
import argparse
# if os.path.isfile(os.path.join(data_path,'training_set_with_missing_images.dat')):
#   print("Part of compressed training set already exists. Removing... ")
#   os.remove(os.path.join(data_path,'training_set_with_missing_images.dat'))

########
# PATHS
########
parser = argparse.ArgumentParser()
parser.add_argument("--data_path",default="/home/wenxuan/book dataset",help="dir that contains the .dat file")
parser.add_argument("--compress_image",default=False)
parser.add_argument("--calc_stats",default=True)
args = parser.parse_args()
data_path = args.data_path # set path to training set images
# data_path="/content/drive/MyDrive/book dataset"
output_path = os.path.join(data_path,'fid_stats.npz') # path for where to store the statistics
# path to store compressed training dataset
img_size=512
dtype=np.float32
#default path
compress_path=os.path.join(data_path,'test_set_with_missing_images.dat')


inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")


def compress_image(set="df_train.csv",size=1000):
    
    df=pd.read_csv(os.path.join(data_path,set))
    assert len(df)>=size,"requested size is larger than size of the dataset "
    df=df.iloc[:size]
    print(f"load first {len(df)} images only", flush=True)

    
    global compress_path
    if "train" in set:
        compress_path=os.path.join(data_path,'training_set_with_missing_images.dat')
    else:
        compress_path=os.path.join(data_path,'test_set_with_missing_images.dat')
    #change every label in df_train into image path
    concat_path=lambda x:os.path.join(data_path+'/images/images',str(x)+'.jpg')
    df=df[df.columns[0]].apply(concat_path)


    image_list=df.values.tolist()#df.values is numpy array
    images=None;failed_list=[]
    read=0;failed=0;compressed_length=0
    #initialize images 
    if os.path.isfile(compress_path):
        try:
            images=np.fromfile(compress_path,dtype=dtype).reshape(-1,img_size,img_size,3)
        except:
            print('Error! Pre-compressed data shape doesn\'t match currently specified image size. Please delete the pre-compressed file. ')
            exit()
        if images.shape[0]<10000:#loading data over this threshold will cause OOE
            images=[images[i] for i in range(images.shape[0])]
            compressed_length=len(images)
            image_list=image_list[compressed_length:]
        else:
            images=[]
    else:
        images=[]
    print(f"Already compressed {compressed_length} images, continue writing into the file from that checkpoint....")


    #save in batch
    with open(compress_path,mode='wb+') as f:
        for i in tqdm.tqdm(range(len(image_list))):
            #change to float32 from uint8(cv2.imread default) for compatiblity with TF model
            # print(image_list[i],os.path.isfile(image_list[i]))
            image=imread(image_list[i])
            if(image is not None):
                image = cvtColor(resize(image,(img_size,img_size),interpolation=INTER_CUBIC),COLOR_BGR2RGB).astype(np.float32)
                images.append(image)
                read += 1
                #save every 1000 iterations
                if i%1000==0 or i==len(image_list)-1:
                    if i==0:
                        compressed_length+=1
                    images=np.array(images)
                    #write data into npy in batch
                    print("Now saving: ",images.shape)
                    images.tofile(f)
                    print("  ||  Already saved: ", compressed_length+i," images")
                    del images
                    images=[]
            else:
                print("failed:",i)
                failed_list+=[image_list[i]]
                failed+=1

    # images = np.array(np.fromfile(compress_path).reshape(-1,img_size,img_size,3))
    # print("%d images compressed" % len(images))

def calc_stats():
    try:
        images = np.array(np.fromfile(compress_path,dtype=dtype).reshape(-1,img_size,img_size,3))
        print("%d images found and loaded"%len(images))
    except Exception as e:
        print('Encountered exception while loading images: ',e) 


    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("Created!")

    print("calculte FID stats..", end=" ", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print("finished")

#run program 
if args.compress_image:
    compress_image(set='df_test.csv',size=4000)
if args.calc_stats:
    print("-------------------------------------")
    calc_stats()