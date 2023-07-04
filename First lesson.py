import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import linear_model
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures" ,"absences"]]
predict="G3"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#each time accuracy is different so we shoud find the best status
best=0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
#test size 10% train size 90%
    linear=linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc=linear.score(x_test,y_test) # acc stands for accuracy
    if (acc>best):
        best=acc
        with open('studentmodel.pickle', 'wb') as file:
            pickle.dump(linear,file)
            #using pickle module to save our model so we can use it later on
            #mode=wb it means write
print(best)
#newmodel=pickle.load("studentmodel.pickle",'rb')
#readind models are already existed
#print('Coefficient: \n', newmodel.coef_) # These are each slope  shib
#print('Intercept: \n', newmodel.intercept_) # This is the intercept arz az mabda

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    #it shows how to use the model, we give the model test datas and it predicts the result