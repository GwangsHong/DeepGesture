from socket import *
import _thread
import struct
import socket
import numpy as np
from threading import Lock
from models import *
from sklearn import preprocessing
from scipy.signal import resample
import queue
import time
#server param
HOST = ''
PORT = 2030
ADDR = (HOST, PORT)
#sensor param
th = 20
LOWER_BOUND = 450
UPPER_BOUND = 550
n_storage = 128
AVERAGE_SIGNAL = 512
#ox 딥모델 파라미터

BAND_NUM = 12
SENSOR_NUM = 6

ACC_X = 0
ACC_Y = 1
ACC_Z = 2
GRY_X = 3
GRY_Y = 4
GRY_Z = 5

# cp = n_storage // 4  # cp: checkpoint
cp = 0 # cp: checkpoint
ep = cp * 2 + cp  # ep: endpoint


params = [["./train_log/ox_model_c5.hdf5", 5],  #O/X 모델 파라미터
          #["./train_log/direction_model_c5.hdf5" , 5] ,  #좌/우/위/아래 모델 파라미터
          # ["./train_log/run_model_c2.hdf5" , 2],  #달리기 모델 파라미터
          # ["./train_log/jump_model_c2.hdf5" , 2],  #점프 모델 파라미터
          # ["./train_log/rowing_model_c2.hdf5" , 2],  #노젓기 모델 파라미터
          # ["./train_log/tug-of-war_model_c2.hdf5" , 2],  #줄다리기 모델 파라미터
          # ["./train_log/video_action_1_c2.hdf5" , 2],  #동영상 컨텐츠 책갈피 1
          # ["./train_log/video_action_2_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 2
          # ["./train_log/video_action_3_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 3
          # ["./train_log/video_action_4_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 4
          # ["./train_log/video_action_5_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 5
          # ["./train_log/video_action_6_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 6
          # ["./train_log/video_action_7_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 7
          # ["./train_log/video_action_8_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 8
          # ["./train_log/video_action_9_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 9
          # ["./train_log/video_action_10_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 10
          # ["./train_log/video_action_11_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 11
          # ["./train_log/video_action_12_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 12
          # ["./train_log/video_action_13_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 13
          # ["./train_log/video_action_14_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 14
          # ["./train_log/video_action_15_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 15
          # ["./train_log/video_action_16_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 16
          # ["./train_log/video_action_17_c2.hdf5", 2],  # 동영상 컨텐츠 책갈피 17
          # ["./train_log/video_action_18_c2.hdf5", 2]  # 동영상 컨텐츠 책갈피 18
          ]

class model_param:
    def __init__(self,param):

        self.pretrained = param[0]
        self.n_classes = param[1]

        self.model = "CNN_GRU"
        self.mode = "test"
        self.n_input = 6
        self.n_steps = 128
        self.input_type = "raw"
        self.n_layers = 4
        self.n_hidden = 256

class  model_server:
    def __init__(self):
        #딥모델 초기화
        print("Deep model laoding...")
        self.deep_models = []
        for i in range(len(params)):
            model = create_model(model_param(params[i]))
            self.deep_models.append(model)
            print("model [{}]loaded".format(i))
        print("Deep model load complete")

        #데이터 초기화
        # self.data = []
        self.init_data()

        self.serverSocket = socket.socket(AF_INET, SOCK_STREAM)  # 1.소켓을 생성한다.
        self.serverSocket.bind(ADDR)  # 2.소켓 주소 정보 할당
        print('bind')
        #listen 시작
        _thread.start_new_thread(self.come, (), )

        while True:

            try:
                self.predict()
            except Exception as e:
                print(e)
    def init_data(self):
        data = []
        for i in range(SENSOR_NUM* BAND_NUM):
            temp = []
            for x in range(n_storage):
                temp.append(AVERAGE_SIGNAL)
            y = np.array(temp)
            data.append(y)
        self.data = np.array(data)
        self.response_band = []

    def come(self):
        while True:
            try:
                self.serverSocket.listen(1)  # 3.연결 수신 대기 상태
                print('listen...')
                self.conn, self.addr = self.serverSocket.accept()  # (socket object, address info) return
                print('Got connection from', self.addr)
                print('accept...')
                self.get_model_message()
                self.get_data()
            except Exception as e:
                print("Disconnected..")
                _thread.start_new_thread(self.come, (), )
                break;
    def get_model_message(self):
        if self.addr !=None:
            data = self.conn.recv(1)
            msg = data.decode('utf-8')
            self.model_idx = int(msg)
            print("모델번호:" + str(self.model_idx) )

    def get_data(self):
        #count = 0
        while True:
            try:
                rev_size = self.conn.recv(4)
                size = struct.unpack('i', rev_size)
                data = self.conn.recv(size[0])
                msg = data.decode('utf-8')
                msg = msg.split(",")
                msg = np.array(msg,dtype=np.int)
                #1) np array 로 변환해서 stack 쌓기

                self.data = self.data[:, 1:]
                self.data = np.hstack((self.data, np.atleast_2d(msg).T))

            except Exception as e:
                print("recv error...")
                print(e)
                self.init_data()
                break

    def predict(self):
        self.response_band = []
        band_max = []
        #밴드 시그널 비교
        sensor_data = []
        for i in range(BAND_NUM):
            check_pt = self.data[i*SENSOR_NUM+GRY_X,cp]

            if (check_pt < LOWER_BOUND or check_pt > UPPER_BOUND) or (
                    check_pt < LOWER_BOUND or check_pt > UPPER_BOUND) or (
                    check_pt < LOWER_BOUND or check_pt > UPPER_BOUND):
                self.response_band.append(i)
                sensor_data.append(self.data[i * SENSOR_NUM:i * SENSOR_NUM + SENSOR_NUM, :])

        scaled_data = []
        for i in range(len(self.response_band)):
            scaled_x_data = []
            for x in sensor_data[i]:
                scaled_x = preprocessing.scale(x)
                scaled_x = resample(scaled_x, 128)
                scaled_x_data.append(scaled_x)
            scaled_x_data = np.array(scaled_x_data, dtype=np.float32)
            scaled_data.append(scaled_x_data)
        scaled_data = np.array(scaled_data)

        if len(scaled_data) > 0:
            scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], scaled_data.shape[2], 1)
            score = self.deep_models[self.model_idx].predict(scaled_data)
            _thread.start_new_thread(self.send_result, (self.response_band,score), )

    def send_result(self,response_band,score):
        classes = params[self.model_idx][1]
        num_band = len(response_band)
        # c , n_b, b_idx, argmax_score, scores, b_idx, argmax_socre, socres, ...
        str = "{}".format(classes)
        str =str + ",{}".format(num_band)

        for i in range(num_band):
            str = str + ",{}".format(response_band[i])
            top = np.argmax(score[i])
            print("band: {0}:{1}".format(response_band[i], top))
            str =str + ",{}".format(top)
            for j in range(len(score[i])):
                str = str +",{}".format(score[i][j])
        try:
            #print(str)
            self.conn.send(str.encode('utf-8'))
        except Exception as e:
            self.conn.close()
            print(e)
            print("send error...Disconnected..")
            _thread.start_new_thread(self.come, (), )

            self.init_data()
if __name__ == "__main__":
    server = model_server()



