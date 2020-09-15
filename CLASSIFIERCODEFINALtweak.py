from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import chi2, SelectKBest
from sklearn import svm
import numpy as np
import csv

np.set_printoptions(threshold=np.inf)

csvfile=open('PREPROCESSEDDATA2.csv')
csvfile=csv.reader(csvfile)
csvfile=list(csvfile)

opencsv=open('FINALLABELS.csv')
opencsv=csv.reader(opencsv)
opencsv=list(opencsv)

##FOR CROSS VALIDATION
skf=StratifiedKFold(n_splits=10)

documents1=[]

###X DATA
a=0
while a<3000:
    text=''.join(csvfile[a])
    documents1.append(text)
    a+=1

vectorizer=TfidfVectorizer()
vectorizer.fit(documents1)
vector=vectorizer.transform(documents1)
features = vectorizer.get_feature_names()
xdata=vector.todense()

##TESTING:

##train_set=("hello john", "hello boy sad")
##test_set=("hi i am sad", "hello i am sad")
##
##vectorizer.fit(train_set)
##vector2=vectorizer.transform(test_set)
##print(vectorizer.vocabulary_)
##print(vector2.todense())

##TFIDF VALUES FOR THE WHOLE DOCUMENT (THIS IS USELESS HAHA)

##tfidf_highest=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
##
##doc=0
##while doc<3000:
##    feature_index = vector[doc,:].nonzero()[1]
##    tfidf_scores = zip(feature_index, [vector[doc, x] for x in feature_index])
##    
##    list_count=0
##    for w, s in [(features[i], s) for (i, s) in tfidf_scores]: ##s == tf_idf score, i== feature index, w== word
##        if s >(tfidf_highest[list_count][1]):
##            tfidf_highest.insert(list_count,(w,s))
##            tfidf_highest.sort(key=lambda tup: tup[1], reverse=True)
##            tfidf_highest.pop(len(tfidf_highest)-1)
##            print(tfidf_highest)
##            list_count+=1
##
##    doc+=1
##    if doc%500==0:
##        print(doc)

##PROPER CODE FOR GETTING TFIDF FOR EACH DOCU

##  doc=0
##  feature_index = vector[doc,:].nonzero()[1]
##  tfidf_scores = zip(feature_index, [vector[doc, x] for x in feature_index])
##  for w, s in [(features[i], s) for (i, s) in tfidf_scores]: ##s == tf_idf score, i== feature index, w== word
##      print(w,s)

    
###LABELS
ydata=[]
b=0
while b<3000:
    appenddata=''.join(opencsv[b])
    if appenddata=='1':
        appenddata="music related"
    if appenddata=='0':
        appenddata="non-music related"
    ydata.append(appenddata)
    b+=1

selector = SelectKBest(chi2, k=20)
selector.fit(xdata, ydata)
# Get idxs of columns to keep
idxs_selected = selector.get_support(indices=True)
print(idxs_selected)

for i in idxs_selected:
    print(features[i])

##GETTING TFIDF FOR MUSIC RELATED TWEETS
xdata_music=[]
c=0
while c<3000:
        if ydata[c]=="music related":
            xdata_music.append(documents1[c])
        c+=1

vector2=vectorizer.transform(xdata_music)


## tfidf_highest=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]

doc=0
feature_array=[]
while doc<len(xdata_music):
    feature_index = vector2[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [vector2[doc, x] for x in feature_index])
    for w, s in [(features[i], s) for (i, s) in tfidf_scores]: ##s == tf_idf score, i== feature index, w== word
        feature_array.append([w,s])
    doc+=1

top_features_music=[]
counter=0
for w,s in feature_array:

    avg_count=1
    a=0
    total_for_average=s
    while a<len(feature_array):
        if w==feature_array[a][0]:
            total_for_average=total_for_average+feature_array[a][1]
            avg_count+=1
            feature_array.pop(a)
            
        if a==len(feature_array)-1:
            feature_avg=total_for_average/avg_count

        a+=1

    top_features_music.append([w,feature_avg])

    counter+=1
    if counter%200==0:
        print(counter)

top_features_music.sort(key=lambda tup: tup[1], reverse=True)

print("MUSIC RELATED")
print("")
print(top_features_music[:20])


##GETTING TFIDF FOR NON MUSIC RELATED TWEETS
xdata_non_music=[]
c=0
while c<3000:
        if ydata[c]=="non-music related":
            xdata_non_music.append(documents1[c])
        c+=1

vector3=vectorizer.transform(xdata_non_music)

## tfidf_highest=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]

feature_array=[]
doc=0
while doc<len(xdata_non_music):
    feature_index = vector3[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [vector3[doc, x] for x in feature_index])
    for w, s in [(features[i], s) for (i, s) in tfidf_scores]: ##s == tf_idf score, i== feature index, w== word
        feature_array.append([w,s])

    doc+=1
        
##  wrong implementation
##       
##        if w == (tfidf_highest[list_count][0]):
##            s=((tfidf_highest[list_count][1])+s)/2 #avg?
##            tfidf_highest.pop(list_count)
##            tfidf_highest.insert(list_count,(w,s))
##
##        elif s >(tfidf_highest[list_count][1]):
##            tfidf_highest.insert(list_count,(w,s))
##            tfidf_highest.sort(key=lambda tup: tup[1], reverse=True)
##            tfidf_highest.pop(len(tfidf_highest)-1)
##            list_count+=1

## next part is for averaging the tfidf scores, then getting top 10

top_features_nonmusic=[]
counter=0
for w,s in feature_array:

    avg_count=1
    a=0
    total_for_average=s
    while a<len(feature_array):
        if w==feature_array[a][0]:
            total_for_average=total_for_average+feature_array[a][1]
            avg_count+=1
            feature_array.pop(a)
            
        if a==len(feature_array)-1:
            feature_avg=total_for_average/avg_count

        a+=1

    top_features_nonmusic.append([w,feature_avg])

    counter+=1
    if counter%200==0:
        print(counter)

top_features_nonmusic.sort(key=lambda tup: tup[1], reverse=True)

print("NON MUSIC RELATED")
print("")
print(top_features_nonmusic[:20])
print(top_features_music[-10:])
print(top_features_nonmusic[-10:])


##input to easily get tfidf score ng certain word
continue_prompt=input("prompt?")
while continue_prompt=="y":
    non_or_music=input("mus or non?")
    if non_or_music=="mus":
        feature_to_find= input("What word? (MUSIC RELATED): ")
        index_to_find=[i for i, tupl in enumerate(top_features_music) if tupl[0] == feature_to_find]
        print(index_to_find)
        print(top_features_music[index_to_find[0]])
        continue_prompt=input("continue?")
        
    if non_or_music=="non":
        feature_to_find= input("What word? (NONMUSIC RELATED): ")
        index_to_find=[i for i, tupl in enumerate(top_features_nonmusic) if tupl[0] == feature_to_find]
        print(index_to_find)
        print(top_features_nonmusic[index_to_find[0]])
        continue_prompt=input("continue?")

    else:
        continue

##FOR MACHINE LEARNING ALGO
lineargo=svm.LinearSVC()

##VARIABLES FOR METRICS
accuracyfinal=0
f1final=0
cohenfinal=0
confusionfinal=0

for_chi_test_x=[] ##same case as predvsground
for_chi_test_y=[] ##same case as predvsground
predvsground=[] #yung prediction labels vs ground labels, i put it here para di sya umulit per validation

##SPLITTING/ CROSS VALIDATION

for train_index, test_index in skf.split(xdata, ydata):

##    testarray=[]
##
##    for i in test_index:
##        testarray.append(documents1[i])
##
##    vectorizer.fit(documents1)
##    vector2=vectorizer.transform(testarray)
##    testdata=vector2.todense()

        
    x_train=xdata[train_index]
    x_test=xdata[test_index]
    y_train=[]
    y_test=[]

    ##'manually' making a y-label lists since the indeces are arrays which can't be used to index
    ## the ydata list because idk im too tired to study. baka dahil sa dimensions ng array
    
    a=0
    while a<len(train_index):
        y_train.append(ydata[train_index[a]])
        a+=1

    a=0
    while a<len(test_index):
        y_test.append(ydata[test_index[a]])
        a+=1

    ##MODEL           
    x,y=x_train, y_train
    lineargo.fit(x,y)
    y_predict=lineargo.predict(x_test)
    
    b=0
    while b<len(y_predict):
        add_this_to_predvsground=("PREDICT: " + y_predict[b] + " / GROUND: " + y_test[b])
        indexnum=test_index[b]

        print(b, "", add_this_to_predvsground, documents1[indexnum])

        write_to_predict=(add_this_to_predvsground + documents1[indexnum])
        saveFile=open('PREDICTIONSPART2.csv', 'a')
        saveFile.write(write_to_predict)
        saveFile.write('\n')
        saveFile.close()
        
        ##if y_predict[b] != y_test[b]:
           ## print(b, "", add_this_to_predvsground, documents1[indexnum])

        for_chi_test_x.append(xdata[indexnum])
        for_chi_test_y.append(y_test[b])
        
        predvsground.append(add_this_to_predvsground)
        b+=1
        
    ##METRICS
    ##note:
    ## y_test== ground values
    ## y_predict==classifier predictions

    
    accuracy=accuracy_score(y_test, y_predict)
    accuracyfinal+=accuracy
    f1=f1_score(y_test, y_predict, pos_label='music related')
    f1final+=f1
    cohen=cohen_kappa_score(y_test, y_predict)
    cohenfinal+=cohen
    confusion=confusion_matrix(y_test, y_predict)
    confusionfinal+=confusion
    print('')
    print('ACCURACY:', accuracy)
    print('F1 SCORE:',f1)
    print('COHEN\'S KAPPA:',cohen)
    print('CONFUSION MATRIX: \n', confusion) #order: tn, fp, fn, tp

print('')
print('ACCURACY:', accuracyfinal/10)
print('F1 SCORE:',f1final/10)
print('COHEN\'S KAPPA:',cohenfinal/10)
print('CONFUSION MATRIX: \n', confusionfinal/10) #order: tn, fp, fn, tp


for_chi_test_x_array= np.squeeze(np.asarray(xdata))
scores, pval=(chi2(for_chi_test_x_array, ydata))


pvalue_continue= True
while pvalue_continue==True:
    try:
        pvalue_evaluation=input("what word")
        pvalue_index=features.index(pvalue_evaluation)
        print("SCORE:", scores[pvalue_index])
        print ("PVALUE:", (pval[pvalue_index]))

    except:
        pass


print('Pvalue:')
pval_top=[]
for i in idxs_selected:
    pval_add=(features[i], scores[i])
    pval_top.append(pval_add)
    print(pval_add)
    print(pval[i])


##THE CODE BELOW IS FOR INPUTTING SOMETHING FOR PREDICTION
##trial_data=[input("Write a something to be classified: ")]
##vector4=vectorizer.transform(trial_data)
##trial_data_tfidf=vector4.todense()
##print(lineargo.predict(trial_data_tfidf))

