
#[Super SloMo]
##High Quality Estimation of Multiple Intermediate Frames for Video Interpolation
## 
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import dataloader
from math import log10
import datetime
from tensorboardX import SummaryWriter
from torchsummary import summary
from monai.metrics.regression import SSIMMetric
from matplotlib import pyplot as plt
import numpy as np
import os
import statistics

dt_now = datetime.datetime.now()
print(dt_now)

# For parsing commandline arguments
# コマンド入力
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help='path to dataset folder containing train-test-validation folders') #trainとtestとvalidationが入っているフォルダのパス
parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to folder for saving checkpoints') #学習したパラメータを保存するフォルダのパス
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model') 
parser.add_argument("--log_dir", type=str, default='log', help='path of log for pretrained model') #学習時のLossやPSNR,SSIMの情報を保存するフォルダのパス
parser.add_argument("--train_continue", type=bool, default=False, help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--width", type=int, required=True, help='width of input image for train')
parser.add_argument("--height", type=int, required=True, help='height of input image for train')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
parser.add_argument("--train_batch_size", type=int, default=6, help='batch size for training. Default: 6.')
parser.add_argument("--validation_batch_size", type=int, default=10, help='batch size for validation. Default: 10.')
parser.add_argument("--init_learning_rate", type=float, default=0.0001, help='set initial learning rate. Default: 0.0001.')
parser.add_argument("--milestones", type=list, default=[100, 150], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]') #学習率を変更するエポック
parser.add_argument("--progress_iter", type=int, default=100, help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.') # validationを行う学習回数(学習回数 = trainのデータ数 / 学習のバッチサイズ)
parser.add_argument("--checkpoint_epoch", type=int, default=5, help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.') #パラメータを保存するエポック
# 画像の正規化に用いるパラメータ
parser.add_argument("--Red", type=float, required=True, help='Training Channel Wise Mean Red')
parser.add_argument("--Green", type=float, required=True, help='Training Channel Wise Mean Green')
parser.add_argument("--Blue", type=float, required=True, help='Training Channel Wise Mean Blue')

args = parser.parse_args()

##[TensorboardX](https://github.com/lanpa/tensorboardX)
### For visualizing loss and interpolated frames

#変数名の定義

writer = SummaryWriter(args.log_dir)

W = args.width
H = args.height
###Initialize flow computation and arbitrary-time flow interpolation CNNs.

# gpu使用設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# model
flowComp = model.UNet(6, 4) #引数：(入力チャンネル数,出力チャンネル数)
flowComp.to(device)

# summary(ch,width,height): モデルの流れを表示
summary(flowComp,(6,W,H))

ArbTimeFlowIntrp = model.UNet(20, 5) #引数：(入力チャンネル数,出力チャンネル数)
ArbTimeFlowIntrp.to(device)

summary(ArbTimeFlowIntrp,(20,W,H)) #入力の型:入力チャンネル数,画像の横幅(width),画像の縦幅(height)

###Initialze backward warpers for train and validation datasets
# backwarp:画像とOptical Flowを用いて、遷移先の画像を予測するメソッドの設定 ex) T=0の画像と、0→tのOptical Flowを用いて、T=tの画像を予測する
trainFlowBackWarp= model.backWarp(W, H, device)
trainFlowBackWarp= trainFlowBackWarp.to(device)

validationFlowBackWarp = model.backWarp(W, H, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)


###Load Datasets


# Channel wise mean calculated on adobe240-fps training dataset
#mean = [0.429, 0.431, 0.397] Adobe240

#Bモードのとき
#Training Channel Wise Mean Red={} 0.18740232644516464
#Training Channel Wise Mean Green={} 0.18619351130619055
#Training Channel Wise Mean Blue={} 0.2058688791034099
#Channel Wise Mean Red={} 0.18590836922168022
#Channel Wise Mean Green={} 0.18448318889299223
#Channel Wise Mean Blue={} 0.20480056187401266

#mean = [0.186, 0.184, 0.205]
# データセット画像のRGBごとの画素値の平均
mean = [args.Red, args.Green, args.Blue]
std  = [1, 1, 1]
# 正規化
normalize = transforms.Normalize(mean=mean,std=std)
# transform: 画像配列をtensor型に変換
# tensor型: GPUによる演算を可能にするための配列構造
transform = transforms.Compose([transforms.ToTensor(), normalize])

# 訓練データの読み込み
trainset = dataloader.SuperSloMo(root=args.dataset_root + '/train', transform=transform, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)

# 検証用データの読み込み
validationset = dataloader.SuperSloMo(root=args.dataset_root + '/validation', transform=transform, train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False)

print(trainset, validationset)


###Create transform to display image from tensor

# 正規化された画像をRGB画像に戻すメソッド
negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

# Lossの遷移を画像で保存する関数
def Loss_Draw(train,val,path):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    x = np.arange(len(train))
    ax.plot(x, train, "-", linewidth=1, label='train')
    x = np.arange(len(val))
    ax.plot(x, val, "-", linewidth=1, label='val')

    ax.legend()
    ax.set_title("Loss Graph")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(0,args.epochs)

    fig.savefig(os.path.join(path,'LossGraph.png'))

# PSNRとSSIMの遷移を画像で保存する関数
def val_Draw(psnr,ssim,path):
    for i in range(2): 
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        if i == 0:
            x = np.arange(len(psnr))
            ax.plot(x,psnr, "-", linewidth=1)
            ax.set_title("PSNR Graph")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('PSNR')
            ax.set_xlim(0,args.epochs)
            fig.savefig(os.path.join(path,'PSNRGraph.png'))
        else:
            x = np.arange(len(ssim))
            ax.plot(x,ssim, "-", linewidth=1)
            ax.set_title("SSIM Graph")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('SSIM')
            ax.set_xlim(0,args.epochs)
            fig.savefig(os.path.join(path,'SSIMGraph.png'))

# 学習率を取得する関数
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


###Loss and Optimizer

# 平均絶対値誤差
L1_lossFn = nn.L1Loss()

# 平均二乗誤差
MSE_LossFn = nn.MSELoss()

params = list(ArbTimeFlowIntrp.parameters()) + list(flowComp.parameters())

# Lossの最適化
optimizer = optim.Adam(params, lr=args.init_learning_rate)
# scheduler to decrease learning rate by a factor of 10 at milestones.
# 学習率の変更を行うメソッド
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)


###Initializing VGG16 model for perceptual loss
vgg16 = torchvision.models.vgg16(pretrained=True)
# vgg16の畳み込み4層目の最終出力(ch:512)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)
summary(vgg16_conv_4_3,(3,W,H))
for param in vgg16_conv_4_3.parameters():
		param.requires_grad = False


### Validation function
# 
def validate():
    # For details see training.
    psnr = 0
    ssim = 0
    tloss = 0
    flag = 1
    with torch.no_grad():
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            # 入力画像
            I0 = frame0.to(device)
            I1 = frame1.to(device)
            # 正解画像
            IFrame = frameT.to(device)
                        
            # U-Net1：入力画像のOptical Flowの予測
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            # Optical Flowの推定に用いる係数を出力
            fCoeff = model.getFlowCoeff(validationFrameIndex, device)

            # 任意時間ステップtへのOptical Flowの推定
            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            # 推定Optical Flowを用いて、時間ステップtのフレームを生成
            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)
            
            # 出力結果を1つの配列に連結
            # cat: 配列の結合
            param=torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1)

            # U-Net2：正確なOptical FlowとVisiblity Mapの予測
            intrpOut = ArbTimeFlowIntrp(param)
                
            # 正確なOptical Flow　※U-Net1で予測したよりも正確なOpical Flow
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1

            # visibility map: 画像内の可視部分の変化領域
            V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1   = 1 - V_t_0
                
            # 正確なOptica Flowを用いて、時間ステップtのフレームを生成
            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
            
            # フレーム生成に使用する係数
            wCoeff = model.getWarpCoeff(validationFrameIndex, device)
            
            # 時間ステップtのフレームを生成(最終出力)
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
            
            # For tensorboard
            if (flag):
                retImg = torchvision.utils.make_grid([revNormalize(frame0[0]), revNormalize(frameT[0]), revNormalize(Ft_p.cpu()[0]), revNormalize(frame1[0])], padding=10)
                flag = 0
            
            
            #loss

            # 再構成損失:生成したフレーム(Ft_p)と参照画像(IFrame)の平均絶対値誤差 
            recnLoss = L1_lossFn(Ft_p, IFrame)

            # 知覚的損失: 学習済みVGG16の重みを生成フレーム(F_tp)と参照画像(IFrame)に適用したものの平均二乗誤差
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
            
            # ワーピング損失
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)
        
            # 滑らかさの損失
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
            
            # 損失関数
            # 係数は経験則
            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
            tloss += loss.item()
            
            #psnr
            MSE_val = MSE_LossFn(Ft_p, IFrame)
            psnr += (10 * log10(1 / MSE_val.item()))
            # SSIM
            data_range = Ft_p.max().unsqueeze(0)
            s = SSIMMetric(data_range)._compute_metric(Ft_p,IFrame)
            s = s.mean().item()
            ssim += s

    return (psnr / len(validationloader)), (ssim / len(validationloader)), (tloss / len(validationloader)), retImg

### /Initialization


# 続きから学習を行う場合の処理
if args.train_continue:
    dict1 = torch.load(args.checkpoint)
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])
else:
    dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'valSSIM': [], 'epoch': -1}


### Training

import time

# tensorboard用のLoss,validation Loss,PSNR,SSIM
start = time.time()
cLoss   = dict1['loss']
valLoss = dict1['valLoss']
valPSNR = dict1['valPSNR']
valSSIM = dict1['valSSIM']
checkpoint_counter = 0

# エポックごとのLoss,validation Loss,PSNR,SSIM
loss_epoch = []
vloss_epoch = []
psnr_epoch = []
ssim_epoch = []

### Main training loop
for epoch in range(dict1['epoch'] + 1, args.epochs):
    print("Epoch: ", epoch)
        
    # Append and reset
    cLoss.append([])
    valLoss.append([])
    valPSNR.append([])
    valSSIM.append([])
    iLoss = 0
    
    # Increment scheduler count    
    scheduler.step()
    
    for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
        
		## Getting the input and the target from the training set
        frame0, frameT, frame1 = trainData
        
        # 入力画像
        I0 = frame0.to(device)
        I1 = frame1.to(device)
        # 参照画像
        IFrame = frameT.to(device)
        
        # 勾配をゼロにする
        optimizer.zero_grad()
        
        # U-Net1：入力画像のOptical Flowの予測
        # Calculate flow between reference frames I0 and I1
        flowOut = flowComp(torch.cat((I0, I1), dim=1))
        # Extracting flows between I0 and I1 - F_0_1 and F_1_0
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]
        
        # Optical Flowの推定に用いる係数を出力
        fCoeff = model.getFlowCoeff(trainFrameIndex, device)
        
        # Calculate intermediate flows:任意時間ステップtへのOptical Flowの推定
        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
        
        # Get intermediate frames from the intermediate flows:推定Optical Flowを用いて、時間ステップtのフレームを生成
        g_I0_F_t_0 = trainFlowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)
        

        # U-Net2：正確なOptical FlowとVisiblity Mapの予測
        # Calculate optical flow residuals and visibility maps
        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
        
        # Extract optical flow residuals and visibility maps
        # 正確なOptical Flow　※U-Net1で予測したよりも正確なOpical Flow
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        # visibility map: 画像内の可視部分の変化領域
        V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0
        
        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)
        
        # フレーム生成に使用する係数
        wCoeff = model.getWarpCoeff(trainFrameIndex, device)
        
        # Calculate final intermediate frame:時間ステップtのフレームを生成(最終出力)
        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
        
        # Loss:損失関数の計算
        recnLoss = L1_lossFn(Ft_p, IFrame)
            
        prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
        
        warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)
        
        loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
        loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
        loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
          
        # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
        # since the loss in paper is calculated for input pixels in range 0-255
        # and the input to our network is in range 0-1
        loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        iLoss += loss.item()
        
        # Validationデータへの適用
        # Validation and progress every `args.progress_iter` iterations
        if ((trainIndex % args.progress_iter) == args.progress_iter - 1):
            end = time.time()
            
            psnr, ssim, vLoss, valImg = validate()
            
            valPSNR[epoch].append(psnr)
            valSSIM[epoch].append(ssim)
            valLoss[epoch].append(vLoss)
            
            #Tensorboard
            itr = trainIndex + epoch * (len(trainloader))
            
            writer.add_scalars('Loss', {'trainLoss': iLoss/args.progress_iter,
                                        'validationLoss': vLoss}, itr)
            writer.add_scalar('PSNR', psnr, itr)
            writer.add_scalar('SSIM', ssim, itr)
            
            writer.add_image('Validation',valImg , itr)
            #####
            
            endVal = time.time()
            
            print(" Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValSSIM: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (iLoss / args.progress_iter, trainIndex, len(trainloader), end - start, vLoss, psnr, ssim, endVal - end, get_lr(optimizer)))
            
            
            cLoss[epoch].append(iLoss/args.progress_iter)
            iLoss = 0
            start = time.time()
    
    # Create checkpoint after every `args.checkpoint_epoch` epochs
    if ((epoch % args.checkpoint_epoch) == args.checkpoint_epoch - 1):
        dict1 = {
                'Detail':"End to end Super SloMo.",
                'epoch':epoch,
                'timestamp':datetime.datetime.now(),
                'trainBatchSz':args.train_batch_size,
                'validationBatchSz':args.validation_batch_size,
                'learningRate':get_lr(optimizer),
                'loss':cLoss,
                'valLoss':valLoss,
                'valPSNR':valPSNR,
                'valSSIM':valSSIM,
                'state_dictFC': flowComp.state_dict(),
                'state_dictAT': ArbTimeFlowIntrp.state_dict(),
                }
        torch.save(dict1, args.checkpoint_dir + "/SuperSloMo" + str(checkpoint_counter) + ".ckpt")
        checkpoint_counter += 1
    
    # 現エポックにおけるLoss,PSNR,SSIMの平均を計算し、保存
    loss_epoch.append(iLoss)
    if len(valLoss[epoch]) != 0:
        vloss_epoch.append(statistics.mean(valLoss[epoch]))
    if len(valPSNR[epoch]) != 0:
        psnr_epoch.append(statistics.mean(valPSNR[epoch]))
    if len(valSSIM[epoch]) != 0:
         ssim_epoch.append(statistics.mean(valSSIM[epoch]))

# matplotlibによるエポックごとのLoss,PSNR,SSIMのグラフ生成
Loss_Draw(loss_epoch, vloss_epoch, args.log_dir)
val_Draw(psnr_epoch, ssim_epoch, args.log_dir)


dt_now = datetime.datetime.now()
print(dt_now)
