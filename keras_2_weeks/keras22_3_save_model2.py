#훈련한 모델 끌어오기
#22_1 카피



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes

#1 데이터
datasets = load_diabetes()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']


x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape, y.shape) #()


x_train, x_test, y_train, y_test = train_test_split(x, y,                                                  #데이터 분할 
    train_size = 0.75,
    random_state = 55,
    shuffle=True)
    
    
print("x값:", x_train.shape, x_test.shape)    #(331, 10) (111, 10) 
print("y값:", y_train.shape, y_test.shape)    #(331,) (111,)
# exit()

#2 모델
model = Sequential()
model.add(Dense(100, input_shape=(10,)))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

# model.summary()

# model.save("./_save/keras22_1_save_model.keras") #상대경로방식 #위치상 훈려전 초기모델

# exit()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 16)  


# model.save("./_save/keras22_3_save_model.keras") #상대참조방식  훈련한 가중치 저장
# model.save("c:/nsu_work/_save/keras22_2_save_model.keras") #절대참조방식

exit()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)      
print("loss = ", loss )                #loss=오차=에러=cost

y_predict = model.predict([x_test])      #100번째 w를 이용해서 y=wx+b에 대입한거임

print("y_test의 원값 : ", y_test)
print("[x_test]의 예측값 : ", y_predict)

r2 = r2_score(y_test,y_predict)        #r2 함수에는 원값과 예측값을 넣으면 됨
print("r2_score :", r2)


rmse = root_mean_squared_error(y_test, y_predict)   #rmse 함수에도 원값과 예측값을 넣으면 됨
print("rmse :", rmse)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

