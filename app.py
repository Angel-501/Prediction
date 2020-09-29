import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl','rb'))

@app.route('/')
def runit():
    return render_template('index.html')
    

@app.route('/impact')
def impact():
    return render_template('entire_impact.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        data1= request.form['id']
        data2= request.form['countupdate']
        data3= request.form['createdby']
        data4= request.form['updatedby']
        data5= request.form['supportincharge']
        data6= request.form['docknowledge']
        data7= request.form['confirmationcheck']
    
        arr= np.array([[data1,data2,data3,data4,data5,data6,data7]])
        pred=model.predict(arr)
        output=round(pred[0])
        print(output)
        return render_template("index.html", output = output)
    else:
        return render_template("index.html")
    
    
    
@app.route('/prediction',methods=['POST'])
def prediction():
    if request.method == 'POST':
        entry1= request.form['id']
        entry2= request.form['active']
        entry3= request.form['count_reassign']
        entry4= request.form['count_opening']
        entry5= request.form['count_updated']
        entry6= request.form['ID_caller']
        entry7= request.form['opened_by']
        entry8= request.form['Created_by']
        entry9= request.form['updated_by']
        entry10= request.form['location']
        entry11= request.form['category_ID']
        entry12= request.form['user_symptom']
        entry13= request.form['Support_group']
        entry14= request.form['support_incharge']
        entry15= request.form['Doc_knowledge']
        entry16= request.form['confirmation_check']
        entry17= request.form['ID_status']
        entry18= request.form['type_contact']
        entry19= request.form['notify']
        entry20= request.form['problem_id']
        entry21= request.form['change_request']
        arr2= np.array([[entry1,entry2,entry3,entry4,entry5,entry6,entry7,entry8,entry9,entry10,entry11,entry12,entry13,entry14,entry15,entry16,entry17,entry18,entry19,entry20,entry21]])
        pred2=model2.predict(arr2)
        output2=round(pred2[0])
        print(output2)
        return render_template('entire_impact.html', output2 = output2)
    else:
          return render_template('entire_impact.html')
    
       
    
    
    
    
if __name__ == "__main__":
    app.run(debug=False)


#import os
#os.getcwd()
#os.chdir("G:/Impact Prediction")
