
from keras.models import Sequential
from keras.layers.core import Dense, Activation #, Dropout
from keras.layers.recurrent import LSTM #, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D,Dropout,Flatten ,Reshape, Bidirectional, Lambda, GRU, Permute
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import os
import datetime

##Model

# args.data: 데이터셋 위치
# args.epochs: ephoch 개수
# args.lr: 학습율
# args.batch-size: 배치크기
# args.n_input: 입력크기
# args.n_hidden: RNN 및 FCN 히든 노드 개수
# args.n_layers: RNN 레이어 개수
# args.n_steps: 시퀀스 길이
# args.n_classes: 분류 개수
# args.mode: 학습 모드 or 테스트 모드
# args.pretrained: 저장된 모델 위치
# args.m: 모델 이름
#cost function: cross endtorpy
#optimization method: adam
def create_model(args):
    #k: kernel size, s: stride, @: channel
    #conv1: k1x5s1x2@12
    #conv2: k1x3s1x1@24
    #conv3: k1x3s1x1@24
    #conv4: k1x3s1x2@48
    #conv5: k1x3s1x1@48
    #conv6: k1x3s1x2@64
    #conv7: k1x3s1x1@64
    #gru1,2,3,4: h256@16
    #FCN: 256

    hyperparams = [[12,5,2],[24,3,1],[24,3,1],[48,3,2],[48,3,1],[64,3,2],[64,3,1]]
    model = Sequential()

    for param in hyperparams:
        model.add(Convolution2D(filters=param[0], kernel_size=(1, param[1]), strides=(1, param[2]),
                                padding="same",
                                input_shape=(args.n_input, args.n_steps, 1)))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Permute(dims=(2,1,3)))
    model.add(Reshape((model.output_shape[1], model.output_shape[2] * model.output_shape[3])))
    for i in range(args.n_layers - 1):
        model.add(BatchNormalization())
        model.add(GRU(args.n_hidden, return_sequences=True))
    model.add(BatchNormalization())
    model.add(GRU(args.n_hidden, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(args.n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    if not args.pretrained == "":
        model.load_weights(args.pretrained)
    if args.mode == "train":
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        save_path = "./train_log/{}_{}/type({})_model({})_classes({})_hidden({})_RNNLayers({})_steps({})".format(timestamp,args.m,args.input_type, args.model, args.n_classes,args.n_hidden,args.n_layers,args.n_steps)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model.summary()
        # plot_model(model, to_file=os.path.join(save_path,'CNN_model.png'))
        return model, ModelCheckpoint(filepath=os.path.join(save_path,'checkpoint_vloss({val_loss:.5f}).hdf5'),monitor='val_loss',save_best_only=False), save_path
    else:
        return model

