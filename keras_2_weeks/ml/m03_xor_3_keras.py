#xor함수는 단일 퍼셉트론(선 하나 긋기)으로 0과 1을 분리하지 못하기때문에 다층 퍼셉트론이 개발되었다.

import numpy as np
from sklearn.svm import LinearSVC #옛날모델. 단층구성
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x_data = np.array([[0, 0], [0, 1],[1,0],[1,1]]) #(4, 2) (4,)
y_data = np.array([0, 1, 1, 0])  # 단층 퍼셉트론 구조, xor

# print(x.shape, y.shape)

#2 모델
# model = Perceptron()
model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(500))
model.add(Dense(510))
model.add(Dense(50))
model.add(Dense(1,activation='sigmoid'))

#3 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['acc'])
 
model.fit(x_data,y_data, batch_size=1,epochs=100,verbose=0) # 다 정해져있어서 epoch도 안함

#4. 평가, 예측
results = model.evaluate(x_data,y_data)
y_predict = np.round(model.predict(x_data))
# result = model.score(x_data,y_data) # accuracy
# print(results)

acc = accuracy_score(y_predict,y_data)
print("acc = ", acc)




