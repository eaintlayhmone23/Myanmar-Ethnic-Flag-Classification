
from flask import Flask,render_template,request,url_for,redirect
import tensorflow as tf
import os
import h5py
import numpy as np
from urllib.request import urlopen
import uuid
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import tensorflow.keras.preprocessing.image as image
from werkzeug.utils import secure_filename

app=Flask(__name__)

flag_classes=["Kachin","Kayah","Karen","Chin","Mon","Burmese","Rakhine","Shan"]
app.secret_key="secret key"
model=load_model('resnet50_e100.h5')
print('Model loaded')

def predict_flag(image_path,model):
    test_image = image.load_img(image_path,target_size = (512,512))
    test_image = image.img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    print(test_image.shape)
    flag_prediction=model.predict(test_image)
    predict_index=int(np.argmax(flag_prediction))
    predict_val=flag_classes[predict_index]
    prob=round(flag_prediction[0][predict_index]*100)
    predict_val='prediction: '+predict_val+'<br>'+'probability: '+str(prob)+'%'
    print(predict_val)
    return predict_val
    

@app.route('/',methods=["GET"])
def home():
    return render_template('index.html')


@app.route('/predict',methods=[ "GET","POST"])
def predict():
    
    
    if request.method == 'POST':
        imagefile=request.files['file']
        if imagefile.filename != "":
            image_path="./upload/" + imagefile.filename
            imagefile.save(image_path)
            result=predict_flag(image_path,model)
            return result
            

        else:
            url = request.form['text']
            resource = urlopen(url)
            unique_filename = str(uuid.uuid4())
            filename = unique_filename+".jpg"
            image_path = os.path.join("./upload/", filename)
            output = open(image_path , "wb")
            output.write(resource.read())
            output.close()
            result=predict_flag(image_path,model)
            return result
            
            
    
        
    return None


if __name__=="__main__":
    app.run(port=5000,debug=True)
    