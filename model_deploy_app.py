import joblib
from tensorflow.keras.models import load_model
import json
from flask import  Flask, request
import pandas as pd
model=load_model(r"insurance (2).h5")
smoker_encoder=joblib.load(r"smoker_encoder (2).pkl")
region_encoder=joblib.load(r"region_encoder (2).pkl")
gen_encoder=joblib.load(r"gen_encoder (8).pkl")


app=Flask(__name__)
@app.route('/', methods=['POST'])

def myfunction():
    data=request.get_json(force=True)
    print(data)
    data=pd.DataFrame([data])
    data['sex']=gen_encoder.transform(data['sex'])
    data['smoker']=smoker_encoder.transform(data['smoker'])
    region_out=region_encoder.transform(data[['region']])
    region_out=pd.DataFrame(region_out.toarray(),columns=['region_northeast', 'region_northwest', 'region_southeast',
       'region_southwest'])
    flatfile=pd.concat([data,region_out],axis='columns')
    flatfile=flatfile.drop('region',axis='columns')
    print(flatfile)
    output=model.predict(flatfile)
    print(output)

    #data=data['info']
    #print(data)
    return str(output)
app.run(host='0.0.0.0',port=5002)