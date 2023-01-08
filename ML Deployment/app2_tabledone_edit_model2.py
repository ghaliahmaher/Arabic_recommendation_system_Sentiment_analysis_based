from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import pandas as pd
import pyarabic.araby as ar
import re , emoji, functools, operator, string
import torch , optuna, gc, random, os
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score , recall_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.data.processors import SingleSentenceClassificationProcessor
from transformers import Trainer , TrainingArguments
# from transformers.trainer_utils import EvaluationStrategy
from transformers.data.processors.utils import InputFeatures
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils import resample
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
import pprint
from collections import Counter
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Giving our application a name --> app
app = Flask(__name__)

# Loading our model and encoder 
load_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/.../final_model/model_final")
load_tokenizer = AutoTokenizer.from_pretrained("C:/Users/.../final_model/model_final")
my_pipeline  = pipeline("sentiment-analysis", model=load_model, tokenizer=load_tokenizer)# outputs a list of dicts 

# This will take you to the home HTML page --> in this example it is called home.html
# Note: in general applications for home pages are usually called --> index.html
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Recommendation', methods=['POST'])
def Recommendation():
    result = my_pipeline(request.form.to_dict()['result'])
    emotion = dict(
        {
            'anger': ['Action','Wrestling','Crime','Thriller'] ,
            'love': ['Romance','Drama','Family'],
            'joy': ['Animation','Comedy','Adventure'],
            'surprise': ['Animation','Thriller','Horror'] ,
            'sadness': ['Drama','Comedy','Family','Animation'],
            'fear': ['Comedy','Family','Animation'],
            'sympathy': ['Drama','Documentary','Biography','Family'] ,
            'bored': ['Animation','Adventure','Thriller','Crime','Sci-Fi'],
            'none': ['Documentary']
        })

    sentiment = {'LABEL_0': "anger",
             'LABEL_1': 'bored',
             'LABEL_2': 'surprise',
             'LABEL_3': 'sadness',
             'LABEL_4': 'fear',
             'LABEL_5' :'sympathy',
             'LABEL_6' :'joy',
             'LABEL_7': 'love',
             'LABEL_8':'none'}
    arabic={'LABEL_0': "غضب",'LABEL_1': 'ملل',
             'LABEL_2': 'تفاجؤ','LABEL_3': 'حزن',
             'LABEL_4': 'خوف','LABEL_5' :'عطف','LABEL_6' :'فرح',
             'LABEL_7': 'حب','LABEL_8': 'بلا شعور'}
    
    x=result[0]['label']
    aarr=result[0]['label']
    
    for key,value in arabic.items():
        if aarr == key:
           aarr = value
           eear=aarr
    
    for key,value in sentiment.items():
        if x == key:
           x = value
           

    dff=pd.read_csv('C:/Users/.../final_model/ML Deployment/clean_stc_data_set.csv')
    
           
    def recommend(x): 
        dic = {}
        result=[]
        for i,n in emotion.items():
            if x == i:
                for j in n:
                    v=dff.loc[dff['program_genre']==j][['original_name','normalized_score']].drop_duplicates(subset = "original_name").sort_values(by='normalized_score',ascending=False)
                    v=v.head(3)
                    for a in v.iterrows():
                        dic[j,a[1][0]]= a[1][1]  
        for key,value in dic.items():
            result.append(key)
        return result
            

    zz=recommend(x) 
    df = pd.DataFrame(zz, columns =['تصنيف الفلم', 'اسم الفلم'])
    urdf=df
    urdf['إعلان الفلم']=urdf['اسم الفلم'].map('https://www.youtube.com/results?search_query=trailer+{}'.format)
    urdf['إعلان الفلم']=urdf['إعلان الفلم'].str.replace(' ','+')
    

    return render_template('Recommendation.html', recommendations_texts= ":الأفلام/المسلسلات المقترحة \n",ee=eear,df=[urdf.to_html(render_links=True, escape=False)],titles=[''])

if __name__ == "__main__":
    app.run(debug=True)