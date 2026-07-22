

# 만약 숫자가 너무크면 어떡하지? 오버플로우 날텐데
# ->전부 소수점으로 만들어버리기- 0~1사이로 수렴
# ->뭐로 나눠? = 데이터에서 가장 큰 값 = MinMaxScaler
# 계산식 = (x-Min)/(Max-Min)
# 이러한 과정을 '정규화' 라고 한다.

#**참고로 물어보는 데이터나 평가도 동일한 shape으로 만들어주어야한다**

#Q. 근데 이거 코드조작 아님?
#A. x데이터 건드는 거고 데이터의 위치,속성은 그대로이다.
# minmax 처리한 x를 학습시키는 거기 때문에 문제는 없다

#단, y는 타겟값이기 때문에 건들면 안된ek
#모든 데이터를 일정한 스케일로 맞추는 거다
#때문에 x에서만 가능하다.

#predict가 범위(0~1)을 벗어나도 이를 맞출 수 있다
# Train_Data를 초기 MinMax 기준으로 잡고
# 벗어난대로 계산할 수 있기 때문에 0이하, 1이상이 있어도 된다


#25 카피


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler#전처리
import numpy as np


#1 데이터
datasets = load_diabetes()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']


x = datasets.data
y = datasets.target
# print(x)
# print(y)

# print(x.shape, y.shape) #()


x_train, x_test, y_train, y_test = train_test_split(x, y,                                                  #데이터 분할 
    train_size = 0.75,
    random_state = 55,
    shuffle=True)
    
    
# print("x값:", x_train.shape, x_test.shape)    #(331, 10) (111, 10) 
# print("y값:", y_train.shape, y_test.shape)    #(331,) (111,)


# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)                                 #model.fit = 훈련을 시키다
x_train = scaler.transform(x_train)     #여기서 0~1 범위 지정
x_test = scaler.transform(x_test)       # 여기선 범위 밖의 값이 나올수도 있음


# print(np.min(x_test), np.max(x_test)) #0.0~1.0   
# print(np.min(x_test), np.max(x_test)) 


# exit()

#2 모델
model = Sequential()
model.add(Dense(100, input_shape=(10,)))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 16)  

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



"""
x_train, x_test, y_train, y_test = train_test_split(x, y,                                                  #데이터 분할 
    train_size = 0.75,
    random_state = 187,
    shuffle=True)

#2 모델
model = Sequential()
model.add(Dense(20, input_shape=(10,)))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 8)  

r2_score : 0.5395573439046958
rmse : 53.02000769962738
mse : 2811.1212164685467



MinMaxScaler 사용

r2_score : 0.533852906015915
rmse : 53.347429943801345
mse : 2845.948281608792

성능 떨어졌누;;

StandardScaler 사용

r2_score : 0.5379272501617807
rmse : 53.113777490942105
mse : 2821.073359357308

MinMax보단 낫네;


"""






