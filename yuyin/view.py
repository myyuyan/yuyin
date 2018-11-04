from django.http import HttpResponse
from django.shortcuts import render
from keras.models import load_model
from python_speech_features import mfcc
import python_speech_features
import numpy as np
from scipy.io import wavfile
import librosa
import scipy
import os
import os
#model=load_model('yuyin.h5')
model=None
choose=True
def index(request):
    return render(request, 'index.html')
def run(request):
    if request.method == 'POST':
        yuyin_0=request.FILES.get('yuyin_0')
        if yuyin_0.name.split('.')[1]!='wav':
            return HttpResponse("文件格式不正确，必须为wav文件!")
        yuyin_0_name='D:\\Github\\python-个人网站\\yuyin\\yuyin\\static\\'+yuyin_0.name
        f=open(yuyin_0_name,'wb')
        for chunk in yuyin_0.chunks():
            f.write(chunk)
        f.close()
        xingbie={1:'女',0:'男'}
        (rate,sig) = wavfile.read(yuyin_0_name)
        mfcc_feat = mfcc(scipy.signal.resample(sig,len(sig) // 2),rate // 2)
        mfcc_feat_div = np.concatenate((mfcc_feat[[0]],mfcc_feat[:-1]))
        mfcc_feat_div_div = mfcc_feat_div -  np.concatenate((mfcc_feat_div[[0]],mfcc_feat_div[:-1]))
        finalfeature = np.concatenate((mfcc_feat,mfcc_feat_div,mfcc_feat_div_div),axis=1)
        xx=[]
        if finalfeature.shape[0]>300:
            xx.append(finalfeature[:300])
        if finalfeature.shape[0]<300:
            xx.append(np.concatenate((finalfeature,np.zeros((300-finalfeature.shape[0],39)))))
        if finalfeature.shape[0]==300:
            xx.append(finalfeature)
        train_x = np.asarray(xx)
        global choose
        global model
        if choose:
            model=load_model('yuyin.h5')
            choose=False
        out=model.predict_classes(train_x)
        code=xingbie[out[0,0]]
        code="发音人的性别为:"+code
        os.remove(yuyin_0_name)
        return HttpResponse(code)