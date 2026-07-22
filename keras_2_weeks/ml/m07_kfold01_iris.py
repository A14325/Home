# 데이터를 0.8, 0.2 로 나누었을 떄
# 데이터를 5등분해서 (T=train, E=test)
# T T T T E
# T T T E T
# T T E T T
# T E T T T 
# E T T T T 

# 모든 등분의 0.2 정도를 test 로 활용하는 것임 -> 매우 좋음
# 단, 훈련량이 5배나 늘어난다

#데이터를 더 잘 쓰기 때문에 효율적이다


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#1. 데이터
x, y = load_iris(return_X_y=True)

Kfold = KFold(n_splits=5, shuffle = True, random_state=123)  # 섞고 5등분 하겠다


#2. 모델
model = DecisionTreeClassifier()

#3. 컴파일, 훈련
#4. 평가, 예측

scores = cross_val_score(model, x, y, cv=Kfold, n_jobs=-1)  #cpu 전체를 쓰겠다

print('acc : ', scores,
      '\n cross_val_score 평균 : ', round(np.mean(scores), 4)
      )

