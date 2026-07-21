

#One_hot_encoding 방식 사용
#다중데이터분류 일때
#output layer = class의 개수
# y를 One_hot_encoding 사용해서 만들기



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_breast_cancer, load_wine  
import time
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping       #얼리스탑핑 임포트한거임**************



#1 데이터
datasets = load_wine()
# print(datasets)
# exit()
# print(datasets.DESCR)
# print(datasets.feature_names)
#sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], 'filename': 'iris.csv', 'data_module': 'sklearn.datasets.data'}
#['sepal length (cm)', 'sepal width (cm)

#exit()

x = datasets.data
y = datasets.target
# print(x)  
# print(y)
# print(x.shape, y.shape) #(178, 13) (178,)
# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48]))   



# #########onehot 1 판다스 이용 ##########
# y = pd.get_dummies(y)
# print (y)
#0,1,2 를 true,false(2차원 행렬)로 변환

########### onehot 2 Tensorflow 이용 #######
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #dumies 쓸때보다 좀더 가독성 있음

# print(y)
 
 
x_train, x_test, y_train, y_test = train_test_split(x, y,                                                  #데이터 분할 
    train_size = 0.75,
    random_state = 333,
    shuffle=True)        
    
    
    
print("x값:", x_train.shape, x_test.shape)    # (133, 13) (45, 13)
print("y값:", y_train.shape, y_test.shape)      #(133, 3) (45, 3) -y값을 onehotencoding 형식으로 바꿔줘야함
print(np.unique(y_train, return_counts=True)) #(array([0., 1.]), array([266, 133]))
exit() 

#2 모델
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(13,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))  # ***다중분류이기에 무조건 softmax = ㅈ대로있는 데이터를  행속 데이터 합이 1이도록 만드는 것
#마지막 레이어 노드=맞춰야할 클래스의 개수(onehot 돌린 y열)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',      #***회귀에서 쓰는 loss가 mse임, 분류에는 이진, 다중 두가지 분류가 있는데, 이진에서는 딱 하나 씀 binary.crossentropy, 
              metrics=['accuracy'],               #이걸 넣으면 메트릭스에 넣은 지표가 로스와 함께 같이 나온다. accuracy는 정확도가 나옴
              )      
es = EarlyStopping(
    monitor='val_loss',     #우린 발로스로 모니터링 할거다
    mode='min',             #최소값으로, 최댓값으로 하고싶으면 max 모르면 auto
    patience=100,           #10번까지 최소값을 갱신 안해도 참아주겠다
    restore_best_weights=True,  #가장 좋은 기록을 가진 모델 가중치를 복원한다
)
start_time = time.time()
model.fit(x_train, y_train, epochs = 100, verbose=0, #행렬형태의 데이터를 주어야 한다
          validation_split=0.1,
          callbacks=[es],)  

end_time = time.time()



print("==================================")
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)      
print("loss = ", loss )                #loss=오차=에러=cost

# exit()


y_predict = model.predict([x_test])  #여기서 나온 데이터는 0에서 1사이의 데이터가 들어가있음, 반올림을 해줘야함
y_predict =  np.argmax(y_test, axis=1) # 가장 큰 값을 1로 지정. 원위치 axis=0(열),1(행)
print(y_predict)

y_test =  np.argmax(y_test, axis=1)
print(y_test)     #y_test와 y_predict 둘을 onehot 하여 둘을 비교
# exit()

print("y_test의 원값 : ", y_test[:5], y_test.shape)
print("[x_test]의 예측값 : ", y_predict[:5], y_predict.shape)


print("걸린시간: ", round(end_time - start_time, 2), "초")

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score :", acc_score)   # accuracy는 0,1,2를 넣어야함->원래값으로 변환해줘야 한다(argmax)


#0,1,2로 되어있는 위치값을 2차원 행렬로 변환하여 계산 하는것
#그 계산값(y_test,y_predict등)을 다시 위치값으로 변환하는 게 argmax 함수(pandas에서)





