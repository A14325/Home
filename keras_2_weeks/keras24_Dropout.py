# 완전연결체. 과연 꼭 해야하는가?-> 파라미터를 '전부' 연결하지않고 솎아낸다
# =DropOut
# *히든레이어를 설명못함.
# 레이어를 늘린다고 무조건 좋아지지도 않음.
# "엥 그러면 꼭 다 연결 안해도되는거 아님?"
# 그래서 몇개 연결안함->연산량 줄어듬. 속도 업. 근데 완전연결과 똑같이 좋아질수도 아닐수도 임.
# 때문에 때에 따라 Dropout을 사용. - 노드 솎아내는 작업
# 모델 재구성과 Dropout의 차이 = "몇 %노드를 이번 훈련에서 제거할 것인가?"

#11_3 카피



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes
import pandas as pd        #판다스를 불러올거고 pd라고 부를거다
import time

#1. 데이터
path = "./_data/"     #그냥 쌩 .은 현재폴더 명시임 그니까 NSU_WORK 폴더라는거, 그리고 data/뒤에가 비어있는 이유는 파이썬 기본 문법에 따라 같은 문자 뒤에 있는걸 그대로 가져올 수 있는데 이 아래 줄에 path + "train.csv"이런게 이 뒤에 붙어있다 치는거임

train_csv = pd.read_csv(path + "train.csv", index_col=0)          #파일 입출력(판다스라는걸 쓸거다), 그리고 0번째 컬럼을 인덱스로 할겁니다(인덱스 = 없는채 친다는것)
test_csv = pd.read_csv(path + "test.csv", index_col=0)  
submit_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)            

#print(submit_csv)
#print(submit_csv.shape)         #(6493, 1)
#print(train_csv)
#print(train_csv.shape)         #(10886, 11)
#print(test_csv)
#print(test_csv.shape)         #(6493, 8)

#print(train_csv.columns)
#Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)      #axis=0은 행방향 1은 열방향
#print(x)                  # [10886 rows x 8 columns]


y = train_csv['count']      #train_csv에서 count만 쏙 빼서 y로 만들겠다
#print(y)
#print(y.shape)      #(10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size= 0.7,
    random_state= 24,           #14 144
    shuffle= True)         #true는 무작위 추출, false는 순차적 추출

#print(x_train.shape, x_test.shape)    #(7620, 8) (3266, 8)
#print(y_train.shape, y_test.shape)    #(7620,) (3266,)

#2. 모델

model = Sequential()
model.add(Dense(250, activation='relu', input_shape=(8,)))
model.add(Dropout(0.2))    #위에 준비한 20개의 노드 중 30%를 랜덤하게 솎아내겠다      
model.add(Dense(500, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. 컴파인, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
model.fit(x_train, y_train, epochs = 200, verbose=2,)
end_time = time.time()                                             #fit 부분의 시간을 재는거

#4. 평가, 예측
y_predict = model.predict(x_test)
rmse = root_mean_squared_error(y_test, y_predict)
print('RMSE =', rmse)                                #RMSE = 157.00259399414062


################ CSV 파일 만들기 #################

y_submit = model.predict(test_csv)    #y_submit은 지금 만들고 학습한 모델로 test_csv를 예측한 데이터이다
#print(y_submit)    
submit_csv['count'] = y_submit      #서브밋 csv 파일에 카운트 부분을 y_submit 데이터로 채울거다
#print(submit_csv)
submit_csv.to_csv(path + "submission_0717_1127.csv") #이제 만든 서브밋csv를 파일로 만들거고, 7월 17일 10시31분에 이파일 만들었다는표시임 그냥

print("걸린시간 : ", round(end_time - start_time, 2), "초")     #round 함수는 소수 몇째자리까지만 보이게 하겠다는거




"""
    model.add(Dense(200, activation='relu', input_shape=(8,)))
    model.add(Dropout(0.3))    #위에 준비한 20개의 노드 중 30%를 랜덤하게 솎아내겠다      
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.3))    
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.3))    
    model.add(Dense(400, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='linear'))
# RMSE = 152.39938354492188
#걸린시간 :  106.35 초

    
    
    
    
"""







