
#04_1 카피

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine  #분류
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeRegressor  # y값이 [0,1,1]=분류   [37.5 측정]=회귀 얘는 회귀, 트리모델
from sklearn.ensemble import RandomForestRegressor 
import warnings 
warnings.filterwarnings("ignore")



#1 데이터
data_list = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True),
    
]

model_list = [
    LinearSVC(),
    LogisticRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),

]


data_name_list = ['아이리스: ', '브래스트캔서: ','와인: ']
model_name_list = ['LinearSVC :', 'LogisticRegression : ', 'DecisionTree : ', 'RF : ']



#2. 모델
for i, value in enumerate(data_list):
    x, y = value
    print("-------------------")
    print(data_name_list[i]) #0번쨰부터 이름 출력해주세요
    print(x.shape, y.shape)

#-------------------
# 아이리스: 
# (150, 4) (150,)
# -------------------
# 브래스트캔서: 
# (569, 30) (569,)
# -------------------
# 와인: 
# (178, 13) (178,)

# exit()

for j, value2 in enumerate(model_list):
    model = value2
    #3 컴파일, 훈련
    model.fit(x,y)  
    #4 평가,예측
    results = model.score(x,y)
    print(model_name_list[j], "model_score :", results)

exit()
