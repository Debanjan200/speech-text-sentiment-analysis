from flask import Flask,render_template,request,redirect
import re,pickle,os
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import numpy as np
import speech_recognition as sr

model=pickle.load(open("model.pkl","rb"))
encoder=pickle.load(open("onehotencoder.pkl","rb"))
cv=pickle.load(open("vectorizer.pkl","rb"))
stemmer=LancasterStemmer()
all_stopwords=stopwords.words("english")
app=Flask(__name__)

def text_extract(text):
    corpus=[]
    emo=re.sub("[^a-zA-Z]",' ',text)
    emo=emo.lower()
    emo=emo.split()
    emo=[stemmer.stem(word) for word in emo if word not in set(all_stopwords)]
    emo=' '.join(emo)
    corpus.append(emo)
    data=cv.transform(corpus)
    predict=model.predict(data)
    lst=[0 for i in range(6)]
    ind=np.argmax(predict)
    lst[ind]=1
    return encoder.inverse_transform([lst])[0][0]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/text",methods=['GET',"POST"])
def text_predict():
    if request.method=="POST":
        res=(request.form["text"])
        res=res.lower()
        res=text_extract(res)
        return render_template("text.html",prediction=res)

    return render_template("text.html")

@app.route("/audio",methods=["GET","POST"])
def audio_predict():
    if request.method=="POST":
        cur_dir=os.path.join(os.getcwd(),"uploads")
        f=request.files["audio"]

        f.save(os.path.join(cur_dir,f.filename))
        audio_path=os.path.join(cur_dir,f.filename)
        r=sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data=r.record(source)
            text=r.recognize_google(audio_data)
        res=text_extract(text)
        return render_template("audio.html",prediction=res,text=text)

    return render_template("audio.html")



if __name__=="__main__":
    app.run(debug=True)
