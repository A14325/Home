#만약 파이썬(혹은 언어) 공부를 하고싶다면 = 가장 얇은 책 볼 것(어린이용)



import numpy as np
from sklearn.datasets import fetch_california_housing

x,y = fetch_california_housing(return_X_y=True)
print(x.shape, y.shape) #(20640, 8) (20640,)

#2. 모델구성
# from sklearn.svm import LinearSVC
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeRegressor  # y값이 [0,1,1]=분류   [37.5 측정]=회귀 얘는 회귀, 트리모델
from sklearn.ensemble import RandomForestRegressor 

# model = LinearSVC() #분류라 안된다
# model = LogisticRegression() #이름이 regress지만 분류모델
# model = DecisionTreeRegressor() # 1.0
model = RandomForestRegressor() #Tree모델을 여러개 모아둔 것이기 때문에 상대적으로 오래걸림. 허나 매우좋음 #0.9741685730806281



#3. 컴파일, 훈련
model.fit(x,y)


#4. 평가, 예측
results = model.score(x,y)
print(results)


