#3 컴파일 훈련에서 얼리스탑핑 임포트 했음 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes
import time

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
    random_state = 7,
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
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping       #얼리스탑핑 임포트한거임**************
es = EarlyStopping(
    monitor='val_loss',     #우린 발로스로 모니터링 할거다
    mode='min',             #최소값으로
    patience=1000,           #10번까지 최소값을 갱신 안해도 참아주겠다
    restore_best_weights=True,  #가장 좋은 기록을 가진 모델 가중치를 복원한다
)

start_time = time.time()
model.fit(x_train, y_train, epochs = 10000000,
          validation_split=0.2,          # validation_split=0.2는 발로스를 트레인데이터에서 20% 검증용 데이터를 더 뽑을거다라는 뜻)
          callbacks= [es])    #모델이 훈련하는 도중 특정 조건에 맞춰 동작을 제어하는 거임.


end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)      
print("loss = ", loss )                #loss=오차=에러=cost

y_predict = model.predict([x_test])      #100번째 w를 이용해서 y=wx+b에 대입한거임
# print("y_test의 원값 : ", y_test)
# print("[x_test]의 예측값 : ", y_predict)

r2 = r2_score(y_test,y_predict)        #r2 함수에는 원값과 예측값을 넣으면 됨
print("r2_score :", r2)

# rmse = root_mean_squared_error(y_test, y_predict)   #rmse 함수에도 원값과 예측값을 넣으면 됨
# print("rmse :", rmse)

# mse = mean_squared_error(y_test, y_predict)
# print("mse :", mse)

print("걸린시간: ", round(end_time - start_time, 2), "초")

