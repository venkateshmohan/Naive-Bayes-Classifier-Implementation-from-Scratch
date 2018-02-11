# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:24:13 2018

@author: Venkatesh T Mohan
"""

import csv
import numpy as np
from math import log

#Open a file and read contents and print total number of documents
f=open("train_data.csv", 'r').readlines()
line=f[-1].split(',')
print(line[0])

#Get the category and increment count of documents in each of them
global category
category=[0 for i in range(21)]

f1=open("train_label.csv",'r').readlines()
for line1 in f1:
    category[int(line1)]+=1
#print(category)   

#Printing class priors
print("Class priors") 
for i in range(20):
     print("P(Omega=",i+1,")=",category[i+1]/int(line[0]))  


c=1
w_c=0

#Creating a dictionary for storing a map of word_id and word_counts
word_count=[dict() for i in range(21)]

#Creating a list for storing the word in each category
global word_category
word_category=[0 for i in range(21)]



#Calculating word count in all documents along each newsgroups
with open("train_data.csv","r") as f:
    cnt=category[c]
    for line in csv.reader(f):
       cat=int(line[0])
       w_c= w_c+int(line[2])
       if int(line[1]) in word_count[c]:
           word_count[c][int(line[1])]+=int(line[2])
       else:
           word_count[c][int(line[1])]=int(line[2])  
       if cat>=cnt:
           word_category[c]=w_c   
           c=c+1
           w_c=0
           if c<=20:
               cnt+=category[c]
           #print(count)
           else:
               break;
#print(word_count[2])

#Calculating the vocabulary length

global length                
length=0   

       
f3=open("vocabulary.txt","r").readlines()
for word in f3:
    length=length+1

c=1
ans=0
b=1
act=1
cnt=0
confuse_matrix=[[0 for a in range(21)] for b in range(21)]
be=[0 for i in range(21)]
#mle=[0 for i in range(21)]
global total
total=0


#Calculating total number of documents in train data
with open("train_label.csv","r") as f4:
    for line in f4:
        total+=1

#Calculating bayesian and maximum likelihood estimates as well as confusion matrix  for train data

doc_count=1   
with open("train_data.csv","r") as f:
     cnt=category[c]
     for line in csv.reader(f):
        b=int(line[0])
        if b==doc_count:
            doc_count+=1
            if b>cnt:
               c=c+1 
               cnt+=category[c]
            act=c
            for i in range(1,21):    
              be[i]=be[i] + log(category[i]/total)
              #mle[i]=mle[i] + log(category[i]/total)
            confuse_matrix[act][np.argmax(be[1:21])+1]+=1 
            #print(conf_matrix[actual][np.argmax(mle[1:21])+1])
            be=[0 for i in range(21)]   
        
        w_id=int(line[1])
        for i in range(1,21):
            if w_id in word_count[i]:
                nk=word_count[i][w_id]
            else:
                nk=0
            be[i]= be[i] + log((nk + 1)/(word_category[i] + length))
            #mle[i]=mle[i]+ log(nk/word_category[i])
            #print(log(nk/word_category[i]))
        
            #print(log((nk + 1)/(word_category[i] + length)))
           
            
              #print(log(category[i]/total))
              #np.argmax(mle[i])      
#print(conf_matrix)
              
#Printing confusion matrix using Bayesian estimates for train data  
print("Confusion matrix for Bayesian estimates: \n")              
from time import sleep              
for i in range(21):
    for j in range(21):
        print('{:4}'.format(confuse_matrix[i][j]),end=' ', flush=True)
        sleep(0)
    print('\n')

#Finding the correctly classified documents
x=np.amax(confuse_matrix,axis=1)
#print(x) 

#Summing up the Correctly classified documents
ct=0
for i in range(21):
    ct=ct+x[i]   
    
#Printing Overall Accuracies and class accuracies of test data using Bayesian Estimates     
print("Overall accuracy:",(ct/total)*100,"%")       
print("Group Accuracies:")
for i in range(21):
    print("Group",i,":",(x[i]/category[i])*100,"%")

    

    
       
     #print(np.argmax(mle))     
        
            
        
        
    


