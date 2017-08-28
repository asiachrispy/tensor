#!/usr/bin/env python
# -*- coding:utf-8 -*-

from PIL import Image
import glob, os
import numpy as np

#ͼƬ������
def timage(folder, new_folder):
    folder_path = glob.glob(folder+ '/*.jpg')
    print folder_path

    for files in folder_path:
        filepath,filename = os.path.split(files)
        filterame,exts = os.path.splitext(filename)

        #�ж�opfile�Ƿ���ڣ��������򴴽�
        if (os.path.isdir(new_folder)==False):
            os.mkdir(new_folder)
        im = Image.open(files)

        #�ж��Ƿ�ΪRGB���ݣ�����Ǻڰ׵ģ�ת��RGB
        image_array = np.array(im)
        #���ڻҶ�ͼû����ɫ��Ϣ��������״Ԫ��ֻ��������ֵ
        if image_array.shape != (224,224,3):
            print "======================================="
            im_RGB = im.convert('RGB')
            #print im_RGB.format, im_RGB.size, im_RGB.mode
            #im_RGB_array = np.array(im_RGB)
            #print im_RGB_array.shape

        im_return = im_RGB.resize((224,224))

        im_return.save(new_folder+'/'+filterame+'.jpg')

if __name__=='__main__':
    #����Ŀ¼
    father_path = '/home/s-20/Image/data/'
    new_father_path = '/home/s-20/Image/train_data/'
    folderList = os.listdir(father_path)

    for folder in folderList:
        children_folder = os.path.join(father_path, folder)
        new_children_folder = os.path.join(new_father_path, folder)
        timage(children_folder, new_children_folder)

    print '�����군��'