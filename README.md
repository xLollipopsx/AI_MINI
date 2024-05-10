# Ex.No: 13 Machine Learning – Mini Project
## DATE: 22/4/2024
## REGISTER NUMBER : 212221060105
# AIM:
To write a program to train the classifier for Diabetes.
## Algorithm:
Step 1: Import packages. Step 2: Get the data. Step 3: Split the data. Step 4: Scale the data. Step 5: Instantiate model. Step 6: Create Gradio Function. Step 7: Print Result.
## Program:
```
import numpy as np
import pandas as pd
pip install gradio
pip install typing-extensions --upgrade
pip install --upgrade typing
pip install typing-extensions --upgrade
import gradio as gr
data = pd.read_csv('/content/diabetes.csv')
data.head()
print(data.columns)
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])
#split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))

def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)
```
## Output:
![WhatsApp Image 2024-05-10 at 11 49 37_e521c476](https://github.com/xLollipopsx/AI_MINI/assets/148363284/9ea21bda-eed4-47cd-9e12-ae2398831593)

![WhatsApp Image 2024-05-10 at 11 49 40_a0796636](https://github.com/xLollipopsx/AI_MINI/assets/148363284/63cfd00d-8153-48ef-972f-dd8f592ee827)

![WhatsApp Image 2024-05-10 at 11 51 15_7d3a5f5a](https://github.com/xLollipopsx/AI_MINI/assets/148363284/07cae06b-e8af-4fa4-841e-a40683dd341b)

## Result:
Thus the system was trained successfully and the prediction was carried out.


