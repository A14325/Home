
#분류 하는 파일

import numpy as np
from sklearn.datasets import load_diabetes

x,y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) #(442, 10) (442,)

exit()
#2. 모델구성
# from sklearn.svm import LinearSVC
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeRegressor  # y값이 [0,1,1]=분류   [37.5 측정]=회귀 얘는 회귀, 트리모델
from sklearn.ensemble import RandomForestRegressor 

# model = LinearSVC() #분류라 안된다
# model = LogisticRegression()  #얘도 Regress 쓰지만 분류다
# model = DecisionTreeRegressor() # 1.0
model = RandomForestRegressor() #Tree모델을 여러개 모아둔 것이기 때문에 상대적으로 오래걸림. 허나 매우좋음 #0.9741685730806281



#3. 컴파일, 훈련
model.fit(x,y)


#4. 평가, 예측
results = model.score(x,y) #원래 분류로 바꿔줘야 하는데 Score 는 자동으로 회귀,분류 가 된다
print(results)


