#12 카피  
#이진분류에서 절대 변하지 않고 사용하는 loss='binary_crossentropy', sigmoid

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_breast_cancer  
import time
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping       #얼리스탑핑 임포트한거임**************


#1 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)


# exit()

x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape, y.shape) #(569, 30) (569,)

print(np.unique(y, return_counts=True))   #(array([0, 1]), array([212, 357]))  #0과 1의 비율을 확인해주는 이진분류 확인법

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,                                                  #데이터 분할 
    train_size = 0.75,
    random_state = 225,
    shuffle=True)
    
    
print("x값:", x_train.shape, x_test.shape)     # (426, 30) (143, 30)
print("y값:", y_train.shape, y_test.shape)     # (426,) (143,)   

# exit()

#2 모델
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(30,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # ***이진분류이기에 무조건 sigmoid(어떤 값을 주던간에 0에서 1사이의 값으로 변경시켜주는 함수) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',      #***회귀에서 쓰는 loss가 mse임, 분류에는 이진, 다중 두가지 분류가 있는데, 이진에서는 딱 하나 씀 binary.crossentropy, 
              metrics=['accuracy'],               #이걸 넣으면 메트릭스에 넣은 지표가 로스와 함께 같이 나온다. accuracy는 정확도가 나옴
              )      
es = EarlyStopping(
    monitor='val_loss',     #우린 발로스로 모니터링 할거다
    mode='min',             #최소값으로, 최댓값으로 하고싶으면 max 모르면 auto
    patience=100,           #10번까지 최소값을 갱신 안해도 참아주겠다
    restore_best_weights=True,  #가장 좋은 기록을 가진 모델 가중치를 복원한다
)
start_time = time.time()
model.fit(x_train, y_train, epochs = 10, batch_size = 32, verbose=1, 
          validation_split=0.1,
          callbacks=[es],)  

end_time = time.time()


print("==================================")
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)      
print("loss = ", loss )                #loss=오차=에러=cost

# exit()


y_predict = model.predict([x_test])      #여기서 나온 데이터는 0에서 1사이의 데이터가 들어가있음, 반올림을 해줘야함
y_predict = np.round(y_predict)          #반올림, accuracy score는 0, 1만 인식하는데, 훈련값은 0~1 사이의 값이기 때문.

print("y_test의 원값 : ", y_test)
print("[x_test]의 예측값 : ", y_predict)

print("걸린시간: ", round(end_time - start_time, 2), "초")

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score :", acc_score)
