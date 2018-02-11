# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:24:13 2018

@author: Venkatesh T Mohan
"""
#Importing variables from Naive Bayes classifier train data file to test data file
from __main__ import word_category,length,total,category
import csv
import numpy as np
from math import log
#Open a file and read contents and print the total number of documents
f=open("test_data.csv", 'r').readlines()
line=f[-1].split(',')
print(line[0])
#Get the category and increment count of documents in each of them
category1=[0 for i in range(21)]
f1=open("test_label.csv",'r').readlines()
for line1 in f1:
    category1[int(line1)]+=1
#print(category)   
    

c1=1
w_c1=0

#Creating a dictionary for storing a map of word_id and word_counts
word_count1=[dict() for i in range(21)]

#Creating a list for storing the word in each category
word_category1=[0 for i in range(21)]

#Calculating word count in all documents along each newsgroups
with open("test_data.csv","r") as f:
    cnt1=category1[c1]
    for line in csv.reader(f):
       cat1=int(line[0])
       w_c1= w_c1+int(line[2])
       if int(line[1]) in word_count1[c1]:
           word_count1[c1][int(line[1])]+=int(line[2])
       else:
           word_count1[c1][int(line[1])]=int(line[2])  
       if cat1>=cnt1:
           word_category1[c1]=w_c1   
           c1=c1+1
           w_c1=0
           if c1<=20:
               cnt1+=category1[c1]
           #print(count)
           else:
               break;
#print(word_count[2])

#Calculating the vocabulary length
length1=0           
f3=open("vocabulary.txt","r").readlines()
for word in f3:
    length1=length1+1

c1=1
ans=0
b1=1
actual1=1
cnt1=0
confuse_matrix2=[[0 for a in range(21)] for b in range(21)]
confuse_matrix1= [[0 for a in range(21)] for b in range(21)]
be1=[0 for i in range(21)]
mle1=[0 for i in range(21)]
total1=0

#Calculating total number of documents in test data
with open("test_label.csv","r") as f4:
    for line in f4:
        total1+=1

#Calculating bayesian and maximum likelihood estimates as well as confusion matrix  for test data
doc_count1=2     
with open("test_data.csv","r") as f:
     cnt1=category1[c1]
     for line in csv.reader(f):
        b1=int(line[0])
        if b1==doc_count1:
            doc_count1+=1
            if b1>cnt1:
               c1=c1+1 
               cnt1+=category1[c1]
            act1=c1
            for i in range(1,21):    
              be1[i]=be1[i] + log(category[i]/total)
              mle1[i]=mle1[i] + log(category[i]/total)
            confuse_matrix1[act1][np.argmax(be1[1:21])+1]+=1
            confuse_matrix2[act1][np.argmax(mle1[1:21])+1]+=1
            
            #print(conf_matrix2[actual1][np.argmax(mle1[1:21])+1])
            be1=[0 for i in range(21)]   
            mle1=[0 for i in range(21)]
        w_id1=int(line[1])
        for i in range(1,21):
            if w_id1 in word_count1[i]:
                nk1=word_count1[i][w_id1]
                if (nk1/word_category[i]) != 0:
                   mle1[i]=mle1[i]+ (log(nk1/word_category[i])*int(line[2])) 
            else:
                nk1=0
            be1[i]= be1[i] + (log((nk1 + 1)/(word_category[i] + length))*int(line[2]))
            
            #print(log(nk/word_category[i]))
                
        
            #print(log((nk + 1)/(word_category[i] + length)))
           
            
              #print(log(category[i]/total))
              #np.argmax(mle[i])      
#print(conf_matrix)
              
#Printing confusion matrix using Bayesian estimates for test data    
print("Confusion matrix for Bayesian estimates: \n")              
from time import sleep              
for i in range(21):
    for j in range(21):
        print('{:4}'.format(confuse_matrix1[i][j]),end=' ', flush=True)
        sleep(0)
    print('\n')

              
#Printing confusion matrix using Maximum likelihood estimates for test data
print("\n \n \n")    
print("Confusion matrix for Maximum likelihood estimates: \n")
from time import sleep             
for i in range(21):
    for j in range(21):
        print('{:4}'.format(confuse_matrix2[i][j]),end=' ', flush=True)
        sleep(0)
    print('\n')    

#Finding the correctly classified documents
x1=np.amax(confuse_matrix2,axis=1)
y=np.amax(confuse_matrix1,axis=1)
#print(x) 

#Summing up the Correctly classified documents
correct_classified1=0    
for i in range(21):
    correct_classified1=correct_classified1+y[i]   
 
correct_classified2=0    
for i in range(21):
    correct_classified2=correct_classified2+x1[i]   
     

#Printing Overall Accuracies and class accuracies of test data using Bayesian Estimates 

print("Overall accuracy:",(correct_classified1/total1)*100,"%")       
print("Group Accuracies:")
for i in range(21):
    print("Group",i,":",(y[i]/category1[i])*100,"%")

    
#Printing Overall Accuracies and class accuracies of test data using Maximum likelihood Estimates 
  
print("Overall accuracy:",(correct_classified2/total1)*100,"%")       
print("Group Accuracies:")
for i in range(21):
    print("Group",i,":",(x1[i]/category1[i])*100,"%")


    

    
       
     #print(np.argmax(mle))     
        
            
        
        
    


