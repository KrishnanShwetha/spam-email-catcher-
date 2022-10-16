
import pandas as pd
import numpy as np
import csv    


def load(path):
    df = None
    
    df = pd.read_csv(path)
    
    #print(df)
   
    return df


def prior(df):
    ham_prior = 0
    spam_prior = 0
    class_id = {}
    #print(df['label'].unique())

    ham_count = 0
    spam_count = 0
    for index, row in df.iterrows(): 
        if row["label"] == 'ham':
            ham_count += 1
        elif row["label"] == 'spam':
            spam_count += 1
    #print("ham_count = ",ham_count)
    #print("spam_count = ",spam_count)
    ham_prior = (ham_count/(ham_count + spam_count))
    spam_prior = (spam_count/(ham_count + spam_count))
   # print("ham_prior = ",ham_prior )
    #print("spam_prior = ",spam_prior )
    
    return ham_prior, spam_prior

def list(row_1):
        words = row_1["text"].split()
        return words

def likelihood(df):

    ham_like_dict = {}
    spam_like_dict = {}
    ham_words = {}
    spam_words = {}
    
    
    for index, row in df.iterrows():
        con_to_list = list(row)
        if (row["label_num"] == 0):
            for word in con_to_list:
                if word not in ham_words: 
                    ham_words[word] = 1
                else: 
                    ham_words[word] = ham_words[word] + 1
    #print("ham_words: ", ham_words)
    
    ham_values = ham_words.values()
    total_ham_words = sum(ham_values)

    #print("total ham = ", total_ham_words)
    
    for word in ham_words:
        likelihood = (ham_words[word]/total_ham_words)
        #print("word = ", word, "; ham_likelihood = ", likelihood)
        ham_like_dict[word] = likelihood
        #print("word =", word, "ham_likelihood",likelihood )
    #print(ham_like_dict)
        
  
    
    for index, row in df.iterrows():
        con_to_list = list(row)
        if (row["label_num"] == 1):
            for word in con_to_list:
                if word not in spam_words: 
                    spam_words[word] = 1
                else: 
                    spam_words[word] = spam_words[word] + 1
    #print("spam_words: ", spam_words)
    
    spam_values = spam_words.values()
    total_spam_words = sum(spam_values)
    #print("total spam = ", total_spam_words)
    
    for word in spam_words:
        likelihood = (spam_words[word]/total_spam_words)
        #print("word = ", word, "; spam_likelihood = ", likelihood)
        spam_like_dict[word] = likelihood
        #print("word =", word, "spam_likelihood",likelihood )
        
    #print(spam_like_dict)
                 
    
    return ham_like_dict, spam_like_dict

def decision(ham_spam_decision,ham_posterior, spam_posterior):
    #print("1.",ham_spam_decision)
    #print("1 if spam, 0 if ham")
    #print("ham_posterior",ham_posterior)
    #print("spam_posterior",spam_posterior)
    if (ham_posterior > spam_posterior):
       # print("2.",ham_spam_decision)
        ham_spam_decision = 1
      #  print("3.",ham_spam_decision)
    elif (ham_posterior < spam_posterior): 
        ham_spam_decision = 0
     #   print("4.",ham_spam_decision)
    #print("ham_spam_decision = ",ham_spam_decision)
    
    return ham_spam_decision
    

def predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, text):
    
    ham_spam_decision = None # 1 if spam, 0 if ham
    
    # posterior = prior * likelihood
    # ham_spam_decision = max[ham_posterior, spam_posterior]
    
    ham_posterior = 1 # posterior probability that it is ham
    spam_posterior = 1 # posterior probability that it is spam
    
    text_words = text.split()

    #print(spam_like_dict)
    #print(ham_like_dict)
    for word in text_words:
        if word not in ham_like_dict:
            ham_posterior = ham_posterior * (0.1)
        else: 
            ham_posterior = ham_like_dict[word]*ham_posterior
        
    ham_posterior2 = ham_prior * ham_posterior
    #print("total ham posterior = ",ham_posterior2)
    #print(" ")
    
    for word in text_words:
        if word not in spam_like_dict:
            spam_posterior = spam_posterior * (0.1)
        else:
            spam_posterior = spam_like_dict[word]*spam_posterior
        #print("word = ", word, "spam_prior = ", spam_prior, "spam_like_dict[word] = ", spam_like_dict[word], "; spam_posterior = ",spam_posterior)
    spam_posterior2 = spam_prior * spam_posterior
    #print("total spam posterior = ",spam_posterior2)
    
    ham_spam_decision = decision(ham_spam_decision,ham_posterior2, spam_posterior2)
    #ham_spam_decision = decision(ham_spam_decision,17, 2)
    
    return ham_spam_decision

def metrics(ham_prior, spam_prior, ham_dict, spam_dict, df):
    #print("asdf")
    #print(df)
    hh = 0 #true negatives, truth = ham, predicted = ham
    hs = 0 #false positives, truth = ham, pred = spam
    sh = 0 #false negatives, truth = spam, pred = ham
    ss = 0 #true positives, truth = spam, pred = spam
    num_rows = df.shape[0]
    for i in range(num_rows):
        roi = df.iloc[i,:]
        roi_text = roi.text
        roi_label = roi.label_num
        guess = predict(ham_prior, spam_prior, ham_dict, spam_dict, roi_text)
        if roi_label == 0 and guess == 0:
            hh += 1
        elif roi_label == 0 and guess == 1:
            hs += 1
        elif roi_label == 1 and guess == 0:
            sh += 1
        elif roi_label == 1 and guess == 1:
            ss += 1
    
    acc = (ss + hh)/(ss+hh+sh+hs)
    precision = (ss)/(ss + hs)
    recall = (ss)/(ss + sh)
    print(acc, precision, recall)
    return acc, precision, recall
            

if __name__ == "__main__":
    val = load("train_1.csv")
    
    hamprior, spamprior = prior(val)
    hamlikedict, spamlikedict = likelihood(val)
    test_stuff = load("/Users/shwethak/Desktop/test.csv")
    metrics(hamprior, spamprior,hamlikedict, spamlikedict, test_stuff)