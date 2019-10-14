#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
from keras import models
from keras import layers
import keras
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import copy
import re


#参数配置
dictLength=6400  #字典长度,根据词频自动省略小频词，只保留5000个词
maxWordlen=5#句子保留的长度精度
embeddingDim=200#词嵌入的维度
discriminator_HiddenSize=64#判别器隐层个数
generator_HiddenSize=64#生成器隐层个数
latent_dim=10#生成器噪声的维度

#显示编码之后的句子
def Show_String_to_tokenString(token,s,codetext):   
    stringList=[]
    stringList.append(s)
    #token字典
    wordDict=token.word_index
    #键值互换字典
    reWordDict={v : k for k, v in wordDict.items()}
    
    codeString=[]
    codeString.append(codetext)
    
    print("原文是：")
    print(stringList[0])
    print("编码文是：")
    for i in range(len(codeString[0])):
        print(reWordDict[codeString[0][i]]+" ",end='')
    print("")


    

#将文本按照词频编码
def getTextFrequencyCode(token,texts):
    #词频编码
    codeTexts=token.texts_to_sequences(texts)
    
    #填充统一长度
    codeTexts=pad_sequences(codeTexts,
                            maxlen=maxWordlen,
                            padding='post',#需要补0时，在序列的起始还是结尾补
                            truncating='post',#当需要截断序列时，从起始还是结尾截断
                            value=0)
    return codeTexts

#读入古诗数据
def getText(path):
    f=open(path,encoding='utf-8')
    Text=f.readlines()
    #分割标点
    for i in range(len(Text)):
        Text[i]=re.split('[，。?《》]',Text[i])
        del(Text[i][-1])
    #保留五言
    Text5=[]
    for i in range(len(Text)):
        for j in range(len(Text[i])):
            if(len(Text[i][j])==5):
                Text5.append(list(Text[i][j]))
                
    return copy.deepcopy(Text5)



if __name__=='__main__':
    print("---start---") 
#    #-------------------------------网络框架--------------------------------------
    #-------------------------------判别器模型------------------------------------
    discriminator_inputs = layers.Input(shape=(5,dictLength),dtype=tf.float32)  
    Word2Vector=layers.Dense(embeddingDim,activation='sigmoid')
    w3=Word2Vector(discriminator_inputs)
    #送入LSTM层
    #x=layers.LSTM(discriminator_HiddenSize,return_sequences=True)(w3)
#    x=layers.LSTM(discriminator_HiddenSize,return_sequences=True)(x)
    x=layers.LSTM(discriminator_HiddenSize,return_sequences=False)(w3)

    
    x=layers.Dense(1, activation='sigmoid')(x)
    discriminator=models.Model(inputs=discriminator_inputs, outputs=x)
    discriminator.summary()
    
    #编译优化器和配置
#    discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0001, clipvalue=1.0, decay=1e-8)
    discriminator_optimizer=optimizers.Adam(lr=0.0001)
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
#     
#    #-------------------------------生成器模型------------------------------------
    generator_input =layers.Input(shape=(5,latent_dim),dtype=tf.float32)
    gLSTM=layers.LSTM(generator_HiddenSize, return_sequences=True,return_state=False)
    x=gLSTM(generator_input)
    y=layers.Dense(dictLength, activation='softmax')(x)
    generator=models.Model(inputs=generator_input, outputs=y)
    generator.summary()
    
##-------------------------------gan模型-------------------------------------
    discriminator.trainable = False
    gan_input =layers.Input(shape=(5,latent_dim),dtype=tf.float32)
    print(generator(gan_input).shape)
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output)
    gan.summary()
    
    #参数配置
#    gan_optimizer = keras.optimizers.RMSprop(lr=0.0001, clipvalue=1.0, decay=1e-8)
    gan_optimizer=optimizers.Adam(lr=0.0003)
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
    
    
#-------------------------------数据获取-------------------------------------
#    #读入数据
#    Text5=getText('minguse.txt')
#    #token统计词频
#    #创建token
#    token = Tokenizer(1+dictLength)
#    #词频统计
#    print("正在统计词频:")
#    token.fit_on_texts(Text5)
#    #返回字典
#    mydic=token.word_index#字典虽然返回全部，但是token.texts_to_sequences时，只取前5000词
#    print("字典大小为%d"%(len(token.word_index))) 
#    #把古诗进行token进行编码
#    codeText=getTextFrequencyCode(token,Text5)
#    Show_String_to_tokenString(token,Text5[10000],codeText[10000])#小测试
#    #把古诗进行one-hot编码，编码后维度为(98270,5,6400)
#    codeText=to_categorical(codeText,dictLength) 
##-------------------------------开始训练-------------------------------------
#    #训练的相关参数
#    iterations = 50000
#    batch_size = 64
#    start = 0
#    a_loss=0
#    d_loss=0
#    
#    for step in range(iterations):
#        #先产生噪声向量
#        random_latent_vectors = np.random.normal(size=(batch_size,5,latent_dim))
#        # 喂入生成器,得到生成文本
#        generated_Text = generator.predict(random_latent_vectors)
#        #-------------------------------训练判别器--------------------------------
#        #batch_size的开始和结束
#        stop = start + batch_size
#        #界限修正
#        if (stop>len(codeText)-1):
#            start=0
#            stop = start + batch_size
#        #混合样本
#        real_text = codeText[start:stop]
#        combined_text = np.concatenate([generated_Text, real_text])
#        #分别给真标签和假标签,假0真1
#        labels = np.concatenate([np.zeros((batch_size, 1)),
#                                 np.ones((batch_size, 1))])
#        # 给标签添加噪声，即不是0、1，而是0.1，0.9
#        for i in range(len(labels)):
#            if i<batch_size:
#                labels[i][0]=labels[i][0]-np.random.uniform(0, 0.1)
#            else:
#                labels[i][0]=labels[i][0]+np.random.uniform(0, 0.1)
#        #训练判别器
#        d_loss = discriminator.train_on_batch(combined_text, labels)
#        
#        #-------------------------------训练GAN-----------------------------------
#        #标签设置为0（真实）用来欺骗判别器
#        misleading_targets = np.ones((batch_size, 1))
#        #训练GAN网络，注意是冻结判别器
#        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
#        
#        #下一轮开始
#        start=start+1
#        
#        
#        
#        
#        #--------------------------------展示成果------------------------------
#        if step % 10 == 0:
#            #输出损失
#            print('第 %s 轮:\n判别器损失: %s\nGAN的损失: %s' % (step, d_loss,a_loss))
#            print("判别器判别生成诗为真的概率:%s"%(discriminator.predict(generated_Text)[0][0]))
#            #显示生成的文本
#            showNum=5
#            for i in range(showNum):
#                #键值互换字典
#                reWordDict={v : k for k, v in mydic.items()}  
#                s0=np.argmax(generated_Text[i][0])
#                if(s0>0):#第一个位置大于0
#                    s0=np.argmax(generated_Text[i][0])
#                    s1=np.argmax(generated_Text[i][1])
#                    s2=np.argmax(generated_Text[i][2])
#                    s3=np.argmax(generated_Text[i][3])
#                    s4=np.argmax(generated_Text[i][4])
#                    
#                    s0=reWordDict[s0]
#                    s1=reWordDict[s1]
#                    s2=reWordDict[s2]
#                    s3=reWordDict[s3]
#                    s4=reWordDict[s4]
#                    print("生成的古诗为：",end='')
#                    print(s0+s1+s2+s3+s4)
#                    print("-----------------------------------------------------")
#    
#    print("---end---")
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
