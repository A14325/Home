
import numpy as np
from sklearn.svm import LinearSVC #옛날모델. 단층구성
from sklearn.metrics import accuracy_score


#1. 데이터
x_data = np.array([[0, 0], [0, 1],[1,0],[1,1]]) #(4, 2) (4,)
y_data = np.array([0, 0, 0, 1]) #and 형태

# print(x.shape, y.shape)

#2 모델
model = LinearSVC()

#3 훈련 / loss가 모델 안에 들어있어서 컴파일 안해도됨
model.fit(x_data,y_data) # 다 정해져있어서 epoch도 안함

#4. 평가, 예측
y_predict = model.predict(x_data)

result = model.score(x_data,y_data) # accuracy
print(result)

acc = accuracy_score(y_predict,y_data)
print("acc = ", acc)