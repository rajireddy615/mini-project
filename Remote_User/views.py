from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import pandas as pd
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,Tweet_Message_model,Tweet_Prediction_model,detection_ratio_model,detection_accuracy_model

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('Search_DataSets')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):


    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Search_DataSets(request):
    if request.method == "POST":
        Tweet_Message = request.POST.get('keyword')
        df = pd.read_csv("./train_tweets.csv")
        df.head()
        offensive_tweet = df[df.label == 1]
        offensive_tweet.head()
        normal_tweet = df[df.label == 0]
        normal_tweet.head()
        # Offensive Word clouds
        from os import path
        from PIL import Image
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        text = " ".join(review for review in offensive_tweet)
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
        fig = plt.figure(figsize=(20, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        # plt.show()
        # distributions
        df_Stat = df[['label', 'tweet']].groupby('label').count().reset_index()
        df_Stat.columns = ['label', 'count']
        df_Stat['percentage'] = (df_Stat['count'] / df_Stat['count'].sum()) * 100
        df_Stat

        def process_tweet(tweet):
            return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split())

        df['processed_tweets'] = df['tweet'].apply(process_tweet)
        df.head()
        # As this dataset is highly imbalance we have to balance this by over sampling
        cnt_non_fraud = df[df['label'] == 0]['processed_tweets'].count()
        df_class_fraud = df[df['label'] == 1]
        df_class_nonfraud = df[df['label'] == 0]
        df_class_fraud_oversample = df_class_fraud.sample(cnt_non_fraud, replace=True)
        df_oversampled = pd.concat([df_class_nonfraud, df_class_fraud_oversample], axis=0)

        print('Random over-sampling:')
        print(df_oversampled['label'].value_counts())
        # Split data into training and test sets
        from sklearn.model_selection import train_test_split
        X = df_oversampled['processed_tweets']
        y = df_oversampled['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=None)
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        count_vect = CountVectorizer(stop_words='english')
        transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
        x_train_counts = count_vect.fit_transform(X_train)
        x_train_tfidf = transformer.fit_transform(x_train_counts)
        print(x_train_counts.shape)
        print(x_train_tfidf.shape)
        x_test_counts = count_vect.transform(X_test)
        x_test_tfidf = transformer.transform(x_test_counts)

        models = []


        # SVM Model
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(x_train_tfidf, y_train)
        predict_svm = lin_clf.predict(x_test_tfidf)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("SVM ACCURACY")
        print(svm_acc)
        models.append(('svm', lin_clf))
        detection_accuracy_model.objects.create(names="SVM", ratio=svm_acc)

        from sklearn.metrics import confusion_matrix, f1_score
        print(confusion_matrix(y_test, predict_svm))
        print(classification_report(y_test, predict_svm))


        #classifier = VotingClassifier(models)
        ##classifier.fit(X_train, y_train)
        #y_pred = classifier.predict(X_test)

        review_data = [Tweet_Message]
        vector1 = count_vect.transform(review_data).toarray()
        predict_text = lin_clf.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Non Offensive or Non Cyberbullying'
        elif prediction == 1:
            val = 'Offensive or Cyberbullying'

        Tweet_Prediction_model.objects.create(Tweet_Message=Tweet_Message,Prediction_Type=val)

        return render(request, 'RUser/Search_DataSets.html',{'objs': val})
    return render(request, 'RUser/Search_DataSets.html')



