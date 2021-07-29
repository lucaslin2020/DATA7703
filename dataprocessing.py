import collections
import csv
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

emails_list = list()
label_list = list()

def data_processing(path):
    files = os.listdir(path)
    for file in files:
        position = path + '\\' + file
        try:
            if 'spam' in file:
                label_sign = 1
            elif 'ham' in file:
                label_sign = 0
            else:
                label_sign = 2
            with open(position, 'r') as f:
                words = f.read()
                lower_words = re.split(r"\W+", words.lower())
                count = collections.Counter(lower_words)
            print(position)

            result = count.most_common(10)
            words_list = list()
            #print('result',result)

            for i in range(len(result)):
                if result[i][0].isalpha():
                    words_list.append(result[i])
            #print('words_list',words_list)

            new_words = ''

            for i in range(len(words_list)):
                for j in range(words_list[i][1]):
                    new_words += words_list[i][0] + ' '
            #print('new_words',new_words)

            emails_list.append(new_words)
            label_list.append(label_sign)
            #print('emails_list',emails_list)
        except Exception:
            os.remove(position)
    count_vector = CountVectorizer()
    words = count_vector.fit_transform(emails_list)
    words_frequency = words.todense()  # 用todense()转化成矩阵
    words_frequency = words_frequency.tolist()
    features = count_vector.get_feature_names()
    return features, words_frequency

def write_data(filename,features,words_frequency):
    with open(filename,"w",encoding="utf-8",newline="") as email_file:
        print('w',filename)
        email = csv.writer(email_file)
        email.writerow(features+['label'])
        for i in range(len(words_frequency)):
            email.writerow(words_frequency[i]+[label_list[i]])

email_path = "C:/uq/uq sem2 2020/DATA7703/Group Project/all_email"
features, frequency = data_processing(email_path)
write_data('all_email.csv',features,frequency)