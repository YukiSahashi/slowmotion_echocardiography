#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import model
import dataloader
import platform
from tqdm import tqdm

from pathlib import Path

import numpy as np
import io
import time
import csv

import uuid
import subprocess

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="", help='path to ffmpeg.exe') # ffmpeg.exeが保存されているフォルダのパス
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model') #学習時に保存したパラメータに関するファイル(.ckpt)のパス　
                                                                                                            #ex)~/SuperSloMo{}.ckpt ※{}には、数字が入る。保存されている中で最も大きい番号のファイルの使用を推奨。
parser.add_argument("--extractDir", type=str, required=True, help='path of extraction') #入力画像が集まったフォルダのパス
parser.add_argument("--gpu", type=int, default=0, help='device number. Default: 0') 
parser.add_argument("--sf", type=int, required=True, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames') #SloMoの倍率　※フレーム数 x sf(2,4,8)
parser.add_argument("--slomo_fps", type=int, required=True, help='frame rate of slomo video') #SloMoの動画のフレームレート
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1') #バッチサイズ
parser.add_argument("--outputDir", type=str, required=True, help='Specify output file name.') #出力先のパス
# 画像の正規化に用いるパラメータ
parser.add_argument("--Red", type=float, default=0.186, help='Training Channel Wise Mean Red')
parser.add_argument("--Green", type=float, default=0.184, help='Training Channel Wise Mean Green')
parser.add_argument("--Blue", type=float, default=0.206, help='Training Channel Wise Mean Blue')

args = parser.parse_args()

# コマンドが正しいか判断する関数
def check():
    error = ""
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    return error


#SuperSloMoで生成した動画をMKVファイル形式で保存する
def create_video(dir,destPath,fps):
    ffmpeg_path = os.path.join(args.ffmpeg_dir, "ffmpeg")
    error = ""
    retn = os.system('{} -y -r {} -i {}/%6d.png -vcodec ffvhuff "{}"'.format(ffmpeg_path, fps, dir, destPath))
    if retn:
        error = "Error creating output video. Exiting."
    return error


def main(extractionDir,slomoFactor,slomo_fps,destDir,gpu):
    # Check if arguments are okay
    error = check()
    if error:
        print(error)
        exit(1)

    #もしPNGのフォルダがあるか
    if os.path.isdir(extractionDir)==False:
        exit(1)

    # 出力フォルダ作成 
    extractionName = os.path.basename(extractionDir)   
    outputPath = os.path.join(destDir, '{}_Slomo_x{}'.format(extractionName, slomoFactor))
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    destPathMKV=os.path.join(destDir,'{}_Slomo_x{}_fps{}.mkv'.format(extractionName, slomoFactor, slomo_fps))

    # ＧＰＵの選択
    if gpu==0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if gpu==1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    # Channel wise mean calculated on adobe240-fps training dataset
    #mean = [0.429, 0.431, 0.397] Adobe240
    # 画像の正規化メソッド
    mean = [args.Red, args.Green, args.Blue]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,std=std)
    # 正規化画像->RGB画像のメソッド
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Load data:画像の読込
    videoFrames = dataloader.Video(root=extractionDir, transform=transform)
    lenFrames = videoFrames.__len__()
    lenMax = lenFrames*slomoFactor
    print('Number of Frames: {}->{}'.format(lenFrames+1,lenMax+1))
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

    # Initialize model:モデルの設定
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    frameCounter = 0

    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):
            I0 = frame0.to(device)
            I1 = frame1.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            # Save reference frames in output folder
            for batchIndex in range(args.batch_size):
                (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, '{:0>6}.png'.format(frameCounter + slomoFactor * batchIndex)))
            
            frameCounter += 1

            # Generate intermediate frames
            for intermediateIndex in range(1, slomoFactor):
                t = float(intermediateIndex) / slomoFactor
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0

                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                for batchIndex in range(args.batch_size):
                    (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, '{:0>6}.png'.format(frameCounter + slomoFactor * batchIndex)))
                frameCounter += 1

            # Set counter accounting for batching of frames
            frameCounter += slomoFactor * (args.batch_size - 1)

            # 生成したフレームの保存
            if frameCounter >= lenMax:
                for batchIndex in range(args.batch_size):
                    (TP(frame1[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, '{:0>6}.png'.format(frameCounter + slomoFactor * batchIndex)))

    # Generate video from interpolated frames:SloMo動画の生成
    create_video(outputPath, destPathMKV ,slomo_fps)

#出力ファイルの存在を確認する
def checkDestFile(outDir,extractionName,sf,fps):

    destMKV='{}_Slomo_x{}_fps{}.mkv'.format(os.path.basename(extractionName), sf, fps)
    destList=os.listdir(outDir)
    for file in destList:
        if (destMKV in file )==True:
            return True
    return False



# 既存ファイルの確認とSloMoのメインに引数を入力
extractionName = os.path.basename(args.extractDir)
print(extractionName)
if checkDestFile(args.outputDir,extractionName, args.sf, args.slomo_fps)==True:
    print("出力先ファイルが存在します。処理済み")
    exit(0)
else:
    print(extractionName)

if extractionName[0] !="_":
    main(args.extractDir, args.sf, args.slomo_fps, args.outputDir, args.gpu)
    
exit(0)

