
#측정자
#목요일에 이상치, 예측치 처리

import numpy as np
from sklearn.datasets import load_diabetes #회귀 데이터셋
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators

#1. 데이터
x,y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) #스케일링 된 x_train 의 구조가 그대로 저장
x_test = scaler.transform(x_test)

#2. 모델
# model = RandomForestRegressor()
all_models = all_estimators(type_filter='regressor') #모든 모델의 리스트
print("all_models :", all_models) 
print("모델의 갯수 :", len(all_models))

#모든 모델을 for 문을 이용해 전부 돌리는 것

# for v in all_models:
#     print(v)               #프린트했을 때 순서가 이름, 알고리즘 순서이기 때문에 밑 for문을 적는다

max_score = 0
max_name = '바보'

for (name, algorithm) in all_models:
    try:   
        model = algorithm()
        
        #3. 훈련
        model.fit(x_train, y_train)
        
        #4. 평가, 예측
        results = model.score(x_test, y_test)
        print(name, '의 정답률 : ', results)
        
        if max_score < results:
            max_score = results
            max_name = name
            
            
    
    except:
        print(name, '은(는) 에러뜬 놈!')
        
    
print("==========================")

print("최고모델 : ", max_name, max_score)

print("==========================")


# ARDRegression 의 정답률 :  0.6100186362674525
# AdaBoostRegressor 의 정답률 :  0.39002699941192875
# BaggingRegressor 의 정답률 :  0.7902265946204095
# BayesianRidge 의 정답률 :  0.6104356531406256
#CCA라는 모델은 'n_components' 를 파라미터로 넣어야 한다 라고 오류가 뜸 -> 얘만 빼기 == try except 





