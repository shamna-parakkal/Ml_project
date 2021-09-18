
from django.shortcuts import redirect, render
from django.http import HttpResponse
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import joblib



def index(request):
    
    return render(request,'index.html')

def predict(request):
    
    Month=request.POST.get('Month')
    Year=request.POST.get('Year')
    new_model = getModel()
    Antibiotic_Sales=new_model.predict([[Month,Year]]).astype(int)
    
    return render(request,'index.html',{'Antibiotic_Sales' : Antibiotic_Sales})

def getModel():
   data=pd.read_csv('Dataset/AntibioticSales.csv')
   data=data.drop(['Date','Time'],axis='columns')
   x=data[['Month','Year']]
   y=data['Sales']
   xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1)
   model=linear_model.LinearRegression()
   model.fit(xtrain,ytrain)
   return model