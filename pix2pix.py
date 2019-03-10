#https://github.com/tommyfms2/pix2pix-keras-byt
#http://toxweblog.toxbe.com/2017/12/24/keras-%e3%81%a7-pix2pix-%e3%82%92%e5%ae%9f%e8%a3%85/
#tommyfms2/pix2pix-keras-byt より


import os
import argparse

import numpy as np

import h5py
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD

from keras.utils import plot_model

import tensorflow as tf
from keras.utils import multi_gpu_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
K.set_session(sess)


import soundfile as sf
from scipy import signal


import models


def normalize(inputs):
    max = inputs.max()
    inputs = inputs / max
    inputs = np.clip(inputs, 0.0, 1.0)
    
    return inputs, max


def log(inputs):
    inputs += 10**-7
    inputs = 20 * np.log10(inputs)
    
    return inputs


def load(segdata_dir, load_number=9999999, complex_input=False):   
    print("data loading\n")
    if complex_input == True or VGG > 0:
        input_dim = 3
    else:
        input_dim = 1
        
    inputs = np.zeros((load_number, input_dim, 256, image_size), dtype=np.float16)
    inputs_phase = np.zeros((load_number, 512, image_size), dtype=np.float16)
    
    datanum = 0
    for i in range(load_number):
        data_dir = segdata_dir + str(i) + "/"
        filelist = os.listdir(data_dir)
        filelist.sort()
        for n in range(len(filelist)):
            if datanum < load_number:
                if filelist[n][-4:] == ".wav" and not filelist[n][0] == "0" and not filelist[n][:-4] == "BGM":           
                    waveform, fs = sf.read(data_dir + filelist[n]) 
                    freqs, t, stft = signal.stft(x=waveform, fs=fs, nperseg=512, 
                                                           return_onesided=False)
                    if not stft.shape == (512,256):
                        stft = stft[:, 1:len(stft.T) - 1]
                    print(stft.shape, datanum)
                    inputs_phase[datanum] = np.angle(stft)
                    if complex_input == True:
                        inputs[datanum][1] = stft[:256].real
                        inputs[datanum][2] = stft[:256].imag
                    inputs[datanum][0] = abs(stft[:256])
                    datanum += 1
    
    if complex_input == True:
        sign = (inputs > 0) * 2 - 1
        sign = sign.astype(np.float16)
        
    inputs = log(inputs)   
    inputs = np.nan_to_num(inputs)
    inputs += 120

    inputs, max = normalize(inputs)

    if complex_input == True:
        inputs = inputs * sign
        
    inputs = inputs.transpose(0, 2, 3, 1)    
    
    if VGG > 0:
        inputs = inputs.transpose(3,0,1,2)
        if VGG == 1:
            inputs[1:3] = 0       # R only
        elif VGG == 3:
            inputs[1] = inputs[0]
            inputs[2] = inputs[0] # Grayscale to RGB
        inputs = inputs.transpose(1,2,3,0)
    
    return inputs, max, inputs_phase


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def plot_generated_batch(X_proc, X_raw, generator_model, batch_size, suffix):
    X_gen = generator_model.predict(X_raw)

    X_raw = X_raw.transpose(0, 3, 1, 2)[0][0]
    X_gen = X_gen.transpose(0, 3, 1, 2)[0][0]  
    X_proc = X_proc.transpose(0, 3, 1, 2)[0][0]


    plt.pcolormesh(X_raw)
    plt.title("original")
    plt.clim(0, 1)
    plt.savefig("./figures/current_batch_"+suffix+"_orig.png")
    plt.show()
    plt.close()

    plt.pcolormesh(X_gen)
    plt.title("prediction")
    plt.clim(0, 1)
    plt.savefig("./figures/current_batch_"+suffix+"_pred.png")
    plt.show()
    plt.close()

    plt.pcolormesh(X_proc)
    plt.title("truth")
    plt.clim(0, 1)
    plt.savefig("./figures/current_batch_"+suffix+"_true.png")
    plt.show()
    plt.close()

       
"""
def restore(Y_true, Y_pred, max, phase, no=0):
    
    Y_pred = Y_pred.transpose(0, 3, 1, 2)
    Y_true = Y_true.transpose(0, 3, 1, 2)    
    
    data_dir = valdata_dir + str(no)
    
    Y_linear = 10 ** ((Y_pred[no][0] * max - 120) / 20)
    Y_linear = np.vstack((Y_linear, Y_linear[::-1]))

    Y_complex = np.zeros((512, image_size), dtype=np.complex128)
    for i in range (512):
        for j in range (image_size):
            Y_complex[i][j] = cmath.rect(Y_linear[i][j], phase[no][i][j])

    Y_Stft = Stft(Y_complex, 16000, label.index[class_n]+"_prediction")
    Y_pred_wave = Y_Stft.scipy_istft()
    Y_pred_wave.plot()
    print(class_n)
    Y_pred_wave.write_wav_sf(dir=pred_dir, filename=None, bit=16)
"""


def extract_patches(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1] // patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2] // patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(procImage, rawImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        # produce an output
        X_disc = generator_model.predict(rawImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1
    else:
        X_disc = procImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc


def my_train(args, X_train, Y_train, X_test, Y_test):
    # load data
    img_shape = X_train.shape[-3:]
    print('img_shape : ', img_shape)
    patch_num = (img_shape[0] // patch_size) * (img_shape[1] // patch_size)
    disc_img_shape = (patch_size, patch_size, Y_train.shape[-1])
    print('disc_img_shape : ', disc_img_shape)

    # train
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_model = models.my_load_generator(img_shape, disc_img_shape)
    #generator_model.load_weights('params_generator_starGAN_epoch_400.hdf5')
    # load discriminator model
    discriminator_model = models.my_load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num)
    #discriminator_model.load_weights('params_discriminator_starGAN_epoch_400.hdf5')
    #generator_model.compile(loss='mae', optimizer=opt_discriminator)                                                                                                                                       
    generator_model.compile(loss='mean_squared_error', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = models.my_load_DCGAN(generator_model, discriminator_model, img_shape, patch_size)

    #loss = [l1_loss, 'binary_crossentropy']                                                                                                                                                                
    loss = ['mean_squared_error', 'binary_crossentropy']

    loss_weights = [1E1, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)
#    plot_model(DCGAN_model, to_file = 'DCGAN.png')
    
    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)
#    plot_model(discriminator_model, to_file = 'discriminator.png')
    
    # start training
    j=0
    print('start training')
    for e in range(epoch):

        starttime = time.time()
        perm = np.random.permutation(X_train.shape[0])
        X_procImage = Y_train[perm]
        X_rawImage  = X_train[perm]
        X_procImageIter = [X_procImage[i:i+batch_size] for i in range(0, X_train.shape[0], batch_size)]
        X_rawImageIter  = [X_rawImage[i:i+batch_size] for i in range(0, X_train.shape[0], batch_size)]
        b_it = 0
        progbar = generic_utils.Progbar(len(X_procImageIter)*batch_size)
        for (X_proc_batch, X_raw_batch) in zip(X_procImageIter, X_rawImageIter):
            b_it += 1
            X_disc, y_disc = get_disc_batch(X_proc_batch, X_raw_batch, generator_model, b_it, patch_size)
            raw_disc, _ = get_disc_batch(X_raw_batch, X_raw_batch, generator_model, 1, patch_size)
            x_disc = X_disc + raw_disc
            # update the discriminator
            disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)

            # create a batch to feed the generator model
            idx = np.random.choice(Y_train.shape[0], batch_size)
            X_gen_target, X_gen = Y_train[idx], X_train[idx]
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            progbar.add(batch_size, values=[
                ("D logloss", disc_loss),
                ("G tot", gen_loss[0]),
                ("G L1", gen_loss[1]),
                ("G logloss", gen_loss[2])
            ])

            # save images for visualization　file名に通し番号を記載して残す
            if b_it % (Y_test.shape[0]//batch_size//2) == 0:
                plot_generated_batch(X_proc_batch, X_raw_batch, generator_model, batch_size, "training" +str(j))
                idx = np.random.choice(Y_test.shape[0], batch_size)
                X_gen_target, X_gen = Y_test[idx], X_test[idx]
                plot_generated_batch(X_gen_target, X_gen, generator_model, batch_size, "validation"+str(j))
                j += 1
        print("")
        print('Epoch %s/%s, Time: %s' % (e + 1, epoch, time.time() - starttime))
        generator_model.save_weights('params_generator_starGAN_epoch_{0:03d}.hdf5'.format(e+1), True)
        discriminator_model.save_weights('params_discriminator_starGAN_epoch_{0:03d}.hdf5'.format(e+1), True)



if __name__=='__main__':    
    segdata_dir = "/home/yui-sudo/document/segmentation/sound_segtest/model_results/2019_0306/Deeplab_75_class_segdata75_256_no_sound/prediction/"
    valdata_dir = "/home/yui-sudo/document/dataset/sound_segmentation/segdata75_256_no_sound/val/"
            
    args = argparse.ArgumentParser(description='Train Font GAN')
    patch_size = 32
    batch_size = 2
    epoch = 500
    load_number = 10
    
    complex_input = False
    VGG = 0                     #0: False, 1: Red 3: White
    image_size = 256
    
    Y_train, max, phase = load(segdata_dir=valdata_dir, load_number=load_number,
                               complex_input=complex_input)

    X_train, max, phase = load(segdata_dir=segdata_dir, load_number=load_number,
                               complex_input=complex_input)
    
    my_train(args, X_train, Y_train, X_train, Y_train)
