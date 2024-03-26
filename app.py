from flask import Flask,request,render_template,send_file
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


app = Flask(__name__)

## Route for a home page
@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            AGE=request.form.get('age'),
            TETSCORE=request.form.get('TETscore'),
            EXPERIENCE=request.form.get('experience'),
            ZONE=request.form.get('zone'),
            LEVEL=request.form.get('level'),
            GENDER=request.form.get('gender'),
            SUBJECT=request.form.get('subject')
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        predict_pipeline.predict(pred_df)

        return render_template('result.html')
        # results=predict_pipeline.predict(pred_df[0])
        # return render_template('home.html',results=results)


@app.route('/result')
def result():
    return render_template('result.html') 


@app.route('/download_excel')
def download_excel():
    filename = 'artifacts/output.xlsx'
    return send_file(filename, as_attachment=True)

if __name__=="__main__":
    app.run(host="0.0.0.0")        
    

