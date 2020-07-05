# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:12:27 2019
2019 0810 only have train and validation train wj+wc val：gd
@author: Dreamen
"""
#%%
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import Sequence
from skimage.transform import resize
import cv2
import math
import SimpleITK
from imgaug import augmenters as iaa
from keras.applications.vgg16 import vgg16,preprocess_input
from keras import layers
from keras.layers import Input, Dropout,Flatten,Dense
#from keras.layers.core import Dense
from keras.models import Model,Sequential
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping,CSVLogger
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import Callback
from lifelines.utils import concordance_index
import sys
sys.path.append("/data/zlw/zhangwenjuan/code")
import resnet
#import scipy.misc
#%%
def calc_at_risk(X, T, O):
    '''
    # function description in survivalnet risklayer
#    tmp = list(T)
#    T = np.asarray(tmp).astype('float64')
    '''
    order = np.argsort(T.astype('float64'))
    sorted_T = T[order]
    at_risk = np.asarray([list(sorted_T).index(x) for x in sorted_T]).astype('int32')
#    T = np.asarray(sorted_T)
    O = O[order]
    X = X[order]
    return X, O, sorted_T, at_risk

class AugmentedImageSequence(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, dataset_csv_file, source_image_dir, batch_size=32,
                 target_size=(224, 224), verbose=0, steps=None,
                 mode='train', random_state=1, flag =0):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param class_names: list of str
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        dataset_df = pd.read_csv(dataset_csv_file)
        self.dataset_df = dataset_df[dataset_df['flag'] ==flag].copy()
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.verbose = verbose
        self.mode = mode
        self.shuffle = True if self.mode=='train' else False
        self.random_state = random_state
        self.prepare_dataset()
        if steps is None:
            self.steps = math.ceil(self.x_path.shape[0] / float(self.batch_size))
        else:
            self.steps = int(steps)
    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(dcm_file, seg_file, ind) for dcm_file, seg_file, ind in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.mode=='train':
            batch_x, *batch_y_ = calc_at_risk(batch_x, batch_y[:, 1], batch_y[:, 0])
            batch_y = np.vstack(batch_y_).T
        else:
            at_risk = np.asarray([list(np.sort(batch_y[:,1])).index(x) for x in batch_y[:,1]], dtype='int32')
            batch_y = np.hstack((batch_y, at_risk.reshape(at_risk.shape+(1,))))
        return batch_x, batch_y
    def load_image(self, dcm_file, seg_file, ind):
        dcm_path = os.path.join(self.source_image_dir, dcm_file)
        seg_path = os.path.join(self.source_image_dir, seg_file)
        dcm = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(dcm_path))[ind, :, :]
        seg = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(seg_path))[ind, :, :]
        #plt.imshow(seg)#
#        (ystart, xstart), (ystop, xstop)= boundingBox( seg, use2D=True)
        pos_x, pos_y = np.where(seg==1)
#        image = dcm[xstart: xstop, ystart:ystop].astype(float)
        image = dcm[pos_x.min():pos_x.max()+1, pos_y.min():pos_y.max()+1].astype(float)
        image_array = (image-image.min()) / (image.max()-image.min())
        image_array = np.stack((image_array, image_array, image_array), -1)
        image_array = cv2.resize(image_array, self.target_size,interpolation=cv2.INTER_CUBIC)
        return image_array

    def transform_batch_images(self, batch_x):
        if self.mode=='train':
            augmenter = iaa.Sequential(
                [
                    iaa.Fliplr(0.3),
                    iaa.Flipud(0.3),
                    iaa.Sometimes(0.3,
                        iaa.SomeOf(1,[
                            iaa.Affine(rotate=(-10,10),cval=0,mode='constant'),
                            iaa.Affine(translate_px=(-10,10),cval=0,mode='constant'),
                            iaa.Crop(percent=(0, 0.1)),
                            iaa.Scale((112,112),
                                    interpolation='cubic',
                                    name=None,
                                    deterministic=False,
                                    random_state=None)
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                        ])),
                ],
                random_order=True,
            )
            batch_x = augmenter.augment_images(batch_x)
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.
        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps*self.batch_size, :]
    def get_image_index(self):
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.x_path[:self.steps*self.batch_size]
    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)

        self.x_path, self.y = df[['dcm_path', 'seg_path', 'slice']].values, df[['event', 'time']].values
    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

class CalCindex(Callback):
   """
   Monitor mean AUROC and update model
   """
   def __init__(self, dataset_csv_file, source_image_dir, batch_size=16,
                 target_size=(224, 224), verbose=0,random_state=1, workers=1, log_path='./'):
      super(Callback, self).__init__()
      self.train_sequence = AugmentedImageSequence(dataset_csv_file, source_image_dir, batch_size=batch_size,
               target_size=target_size, verbose=verbose, steps=None,
               mode='test', random_state=random_state, flag = 0)
      self.val_sequence = AugmentedImageSequence(dataset_csv_file, source_image_dir, batch_size=batch_size,
               target_size=target_size, verbose=verbose, steps=None,
               mode='test', random_state=random_state, flag = 1)
      self.workers = workers
      self.log_path = os.path.join(log_path, 'cindex.log')

   def on_epoch_end(self, epoch, logs={}):
      """
      Calculate the average AUROC and save the best model weights according
      to this metric.

      """
      print("\n*********************************")
        #show learning rate
      self.lr = float(K.eval(self.model.optimizer.lr))
      print(f"current learning rate: {self.lr}")#python 3.6
#        print("current learning rate: %d",% self.lr) for python 3.5
      """
      y_hat shape: (#samples, len(class_names))
      y: [(#samples, 1), (#samples, 1) ... (#samples, 1)]
      """
      cindex = []
      for se in [self.train_sequence, self.val_sequence]:
          y_hat = self.model.predict_generator(se, workers=self.workers)
          y = se.get_y_true()
          path = se.get_image_index()
          pid = np.array([p.split('/')[-2] for p in path[:, 0]])
          res_all = pd.DataFrame(dict(PatientID=pid, Event=y[:, 0], Time=y[:, 1], Risk=y_hat.squeeze()))
          res = pd.DataFrame([[pid_, group.Event.mean(), group.Time.mean(), group.Risk.mean()] for pid_, group in res_all.groupby(by='PatientID')], columns=list(res_all))
          res_all.to_csv(f'..\\icc_csv\\all_{epoch}.csv', index=False)
          res.to_csv(f'..\\icc_csv\\{epoch}.csv', index=False)
          cindex.append(1-concordance_index(res.Time, res.Risk, res.Event))

      print(f"*** epoch#{epoch + 1} dev ***")
      print(f"cindex: {cindex}")
      print("*********************************")
      with open(self.log_path, "a") as f:
          f.write(f"(epoch#{epoch + 1}) cindex: {cindex}, lr: {self.lr}\n")
      return
#%%model 2 resnet 18
def model_icc(input_shape):
#    img_input = Input(shape=input_shape)
    base_model = resnet.ResnetBuilder.build_resnet_18(input_shape, num_outputs=1)
    base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []
    base_model.output_layers = [base_model.layers[-1]]
    x = base_model.layers[-1].output
    x = Dropout(0.5)(x)
    risk_pred = Dense(1, activation="sigmoid", name="risk_pred", use_bias=False)(x)
    model = Model(inputs=base_model.input, outputs=risk_pred)
    return model
# the code is the same as deepsurv
def loss_icc(y_true, y_pred):#y_true：observation
    exp = K.exp(y_pred)[::-1] ##xiaohan origin
    partial_sum = K.cumsum(exp)[::-1]
#    log_at_risk = K.log(K.gather(partial_sum, K.cast(y_true[:, 2], dtype='int32')))
    log_at_risk = K.log(partial_sum)#+ y_pred.max()
    diff = y_pred - log_at_risk
    cost = -K.sum(y_true[:,0]*diff)/(K.sum(y_true[:,0])+1e-6)
    return cost
#%%
if __name__ == '__main__':
    os.chdir("/data/zlw/survival/code")
    output_dir = '../2/h5907'
    source_image_dir = '../data/data3sliceall3'
    dataset_csv_file = '../csv/path3sliceall3Umajor.csv'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    dataset = pd.read_csv(dataset_csv_file)
    model_weights_file = None #'./icc_model/icc_res50.05-11.2735.h5'
    batch_size =24# 8
    lr = 0.001
    #initial_learning_rate = 0.0001
    train_steps = math.ceil(dataset[dataset.flag==0].shape[0] / batch_size)
    input_shape = 224
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    running_flag_file = os.path.join(output_dir, ".training.lock")
    if os.path.isfile(running_flag_file):
        os.remove(running_flag_file)
        #raise RuntimeError("A process is running in this directory!!!")
    else:
        open(running_flag_file, "a").close()
    try:
        model = model_icc((input_shape, input_shape, 3))####if vgg16 deleate (input_shape, input_shape, 3)
        # model = model_icc((input_shape, input_shape, 3))
        # model_weights_file = '/data/zlw/survival/modelweight811/783765.h5'
        if model_weights_file:
           model.load_weights(model_weights_file)
        #print(model.summary())
        print("** create image generators **")
        train_sequence = AugmentedImageSequence(dataset_csv_file, source_image_dir, batch_size=batch_size,
              target_size=(input_shape, input_shape), verbose=1, steps=train_steps, mode='train', flag = 0)
        val_sequence = AugmentedImageSequence(dataset_csv_file, source_image_dir, batch_size=batch_size,
                     target_size=(input_shape, input_shape), verbose=1, steps=None,
                     mode='test', flag = 1)
        # SGD = SGD(lr=initial_learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        Adamop = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        print("** compile model with class weights **")
        model.compile(optimizer=Adamop, loss={'risk_pred':loss_icc})#{'sequential_8':loss_icc}
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(output_dir, 'icc_res18.{epoch:02d}-{val_loss:.4f}.h5'),
            save_weights_only=False,save_best_only=False,verbose=1)
        cal_cindex = CalCindex(dataset_csv_file, source_image_dir, batch_size=batch_size,
                     target_size=(input_shape, input_shape), log_path=output_dir)
        csv_logger = CSVLogger(os.path.join(output_dir, 'traininglog.csv'), separator=',', append=False)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        callbacks = [
                # checkpoint,
                TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
                ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=20,
                                  verbose=1, mode="min",min_lr=1e-5, min_delta=0.0001),
                cal_cindex,
#                early_stopping,
                csv_logger
            ]
        print("** start training **")
        his= model.fit_generator(
                generator=train_sequence,
                steps_per_epoch=train_steps,
                epochs=1000,
                validation_data=val_sequence,
                validation_steps=math.ceil(dataset[dataset.flag==1].shape[0] / batch_size),
                callbacks=callbacks,
                workers=16,
                shuffle=False,
        
            )
    finally:
        os.remove(running_flag_file)