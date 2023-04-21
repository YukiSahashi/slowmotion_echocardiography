import argparse
import os
import os.path
from shutil import rmtree, move, copy
import random

import pydicom
from PIL import Image
import numpy as np
import PIL
from PIL import UnidentifiedImageError
from concurrent import futures
import csv
from natsort import natsorted

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--trainDicoms_folder", type=str, required=True, help='path to the folder containing training dicoms')
parser.add_argument("--testDicoms_folder", type=str, required=True, help='path to the folder containing test dicoms')
parser.add_argument("--dataset_folder", type=str, required=True, help='path to the output dataset folder')
parser.add_argument("--width", type=int, default=640, help='resize image width')
parser.add_argument("--height", type=int, default=640, help='resize image height')
args = parser.parse_args()

# DicomデータをRGB画像に変換
def extractImg(imgYBR,i,left,top,w,h,destPath):
    imgRGB=pydicom.pixel_data_handlers.convert_color_space(imgYBR[i,:,:,],'YBR_FULL_422','RGB')
    pil_img = Image.fromarray(imgRGB)
    im_crop= pil_img.crop((left, top, left+w, top+h))
    im_crop.save('{}\\{:06}.png'.format(destPath,i))

# dicomデータから2心拍分のデータを抜き取る
def extract_frames_dicom(videos, w, h, inDir, outDir):

    for video in videos:
        srcPath=os.path.join(inDir, video)
        destPath=os.path.join(outDir, os.path.splitext(video)[0])

        #１文字目がアンダーバーは使用しない
        if os.path.basename(srcPath)[0] =="_":
            return
        if os.path.exists(destPath) == False:
            os.mkdir(destPath)
    
        ds=pydicom.dcmread(srcPath)

        try:
            imgYBR=ds.pixel_array
        except PIL.UnidentifiedImageError:
            print(srcPath)
            continue

        
        Frames=int( ds[0x0028,0x0008].value )
        EffectiveDuration=float( ds[0x0018,0x0072].value )*1000 #ms
        
        left=ds[0x0028,0x0011].value/2
        top=0
        if "SequenceOfUltrasoundRegions" in ds:
            sq=ds[0x0018,0x6011]        
            l = sq[0].RegionLocationMinX0
            t = sq[0].RegionLocationMinY0
            r = sq[0].RegionLocationMaxX1
            b=  sq[0].RegionLocationMaxY1
            left= (l+r)/2 - w/2
            top = t-10
        else:
            left = left - w/2
            top = 0

        # 2心拍分のデータを抜き取り
        RVec = 0
        if "RWaveTimeVector" in ds:     
            RVec=ds[0x0018,0x6060].value
            phase=range(0,Frames)
            if type(RVec)==list and len(RVec)>3:
                phase=range(int( ( float(RVec[1])/EffectiveDuration)*Frames ),int( ( float(RVec[3])/EffectiveDuration)*Frames ))

        future_list = []
        with futures.ThreadPoolExecutor(max_workers=6) as executor:
            for i in phase:
                future = executor.submit(fn=extractImg, imgYBR=imgYBR, i=i,left=left, top=top, w=w, h=h, destPath=destPath)
                future_list.append(future)
        _ = futures.as_completed(fs=future_list)


# RGB画像を含むをフォルダから12枚ずつ画像を抜き取り、保存
def create_clips(root, destination):
    folderCounter = -1
    files = os.listdir(root)

    # Iterate over each folder containing extracted video frames.
    for file in files:
        images = natsorted(os.listdir(os.path.join(root, file)))
        # print(file)

        for imageCounter, image in enumerate(images):
            # Bunch images in groups of 12 frames
            if (imageCounter % 12 == 0):
                if (imageCounter + 11 >= len(images)):
                    break
                folderCounter += 1
                os.mkdir("{}/{}".format(destination, folderCounter))
            copy("{}/{}/{}".format(root, file, image), "{}/{}/{}".format(destination, folderCounter, image))
        #rmtree(os.path.join(root, file))

def main():
    # Create dataset folder if it doesn't exist already.
    if not os.path.isdir(args.dataset_folder):
        os.mkdir(args.dataset_folder)

    extractTrainPath      = os.path.join(args.dataset_folder, "extractedTrain")
    extractTestPath      = os.path.join(args.dataset_folder, "extractedTest")
    trainPath        = os.path.join(args.dataset_folder, "train")
    testPath         = os.path.join(args.dataset_folder, "test")
    validationPath   = os.path.join(args.dataset_folder, "validation")
    
    if os.path.exists(extractTestPath) == False:
        os.mkdir(extractTestPath)
    if os.path.exists(extractTrainPath) == False:
        os.mkdir(extractTrainPath)
    if os.path.exists(trainPath) == False:
        os.mkdir(trainPath)    
    if os.path.exists(testPath) == False:
        os.mkdir(testPath)    
    if os.path.exists(validationPath) == False:
        os.mkdir(validationPath)
    
    W = args.width
    H = args.height

    # custom dataset        
    # Extract video names
    trainNames = os.listdir(args.trainDicoms_folder)
    testNames =os.listdir(args.testDicoms_folder)

    #Create train-test dataset
    extract_frames_dicom(testNames, W, H, args.testDicoms_folder, extractTestPath)
    create_clips(extractTestPath, testPath)
    extract_frames_dicom(trainNames, W, H, args.trainDicoms_folder, extractTrainPath)
    create_clips(extractTrainPath, trainPath)



    #Select clips at random from test set for validation set.
    # testデータセットからvalidationデータセットを抜き取る
    testClips = os.listdir(testPath)
    indices = random.sample(range(len(testClips)), min(100, int(len(testClips) / 5)))
    for index in indices:
        move("{}/{}".format(testPath, index), "{}/{}".format(validationPath, index))

main()
