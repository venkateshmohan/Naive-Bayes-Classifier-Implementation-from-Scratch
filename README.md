# Naive-Bayes-Classifier-Implementation-from-Scratch
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 diﬀerent newsgroups.
The data is organized into 20 diﬀerent newsgroups, each corresponding to a diﬀerent topic. Here is a list of the 20 newsgroups:
alt.atheism comp.graphics 
comp.os.ms-windows.misc               sci.med
comp.sys.ibm.pc.hardware              sci.space
comp.sys.mac.hardware                 soc.religion.christian
comp.windows.x misc.forsale           talk.politics.guns 
rec.autos                             talk.politics.mideast
rec.motorcycles                       talk.politics.misc 
rec.sport.baseball                    talk.religion.misc
rec.sport.hockey                      sci.crypt 
sci.electronics     

This processed version represents 18824 documents which have been divided to two subsets: training (11269 documents) and testing (7505 documents).
 
There are six ﬁles: map.csv, train label.csv, train data.csv, test label.csv, test data.csv, vocabulary.txt.
The vocabulary.txt contains all distinct words and other tokens in the 18824 documents. 
The train data.csv and test data.csv are formatted "docIdx, wordIdx, count", where docIdx is the document id, wordIdx represents the word id (in correspondence to vocabulary.txt) and count is the frequency of the word in the document.
The train label.csv and test label.csv are simply a list of label id’s indicating which newsgroup each document belongs to. 
The map.csv maps from label id’s to label names.

For each target value ωj (each newsgroup) 
• Calculate class prior P(ωj) 
• Calculate n: total number of words in all documents in class ωj (i.e., total length)
• For each word wk in Vocabulary  
                -Calculate nk: number of times word wk occurs in all documents in class ωj. 
                -Calculate Maximum Likelihood estimator PMLE(wk|ωj) = nk/n 
                            Bayesian estimator PBE(wk|ωj) = nk+1/n+|V ocabulary| (this is Laplace estimate).
