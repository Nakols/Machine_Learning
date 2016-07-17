# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 21:53:12 2016

@author: Nikita
"""
from sklearn.datasets import fetch_20newsgroups # Набор данных - посты-обсуждения новостей определенных категорий
cat = ['talk.religion.misc','soc.religion.christian'] # Выбираем две категории для классификации
# Данные для обучения
twenty_train = fetch_20newsgroups(subset='train', categories = cat, shuffle=True, random_state=1
                                  ,remove=('headers', 'footers', 'quotes'))
# Данне для проверки
twenty_test = fetch_20newsgroups(subset='test', categories = cat, shuffle=True, random_state=1
                                  ,remove=('headers', 'footers', 'quotes'))
# Кусочек данных
print('Text: \n' + str(twenty_train.data[0]) + '\n\nClass: \n' + str(twenty_train.target[0]))
print('Train sample size: ' + str(len(twenty_train.data)))
print('Test sample size: ' + str(len(twenty_test.data)))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Импорт и создание векторайзеров
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizers = {'CountVect': CountVectorizer(binary = False), 'CountVectBin': CountVectorizer(binary = True), 'TFIDFVect': TfidfVectorizer()}
for i in vectorizers:
    print(vectorizers[i])
vectorized_text = {}
for i in vectorizers:
    vectorized_text[i] = vectorizers[i].fit_transform(twenty_train.data)
# csr_matrix - часто используемый эффективный объект для хранения разреженных матриц
# indices  - массив индексов столбцов
# data - массив ненулевых элементов матрицы
# indptr  - массив указателей на начала строк в индексах и данных
import scipy.sparse as sps
csr_matrix_example = sps.csr_matrix([[1, 0, 0], [0, 5, 0], [0, 2, 3]])
print(csr_matrix_example.toarray())
print('indices = ' + str(csr_matrix_example.indices))
print('data = ' + str(csr_matrix_example.data))
print('indptr = ' + str(csr_matrix_example.indptr))

# Переменные с метками классов
train_labels = twenty_train.target
test_labels = twenty_test.target

# Функция пересоздания pandas dataframe с результатами кросс-валидации
def df_auc_rebuild():
    global df_line, df_metrics, df_auc
    df_auc = None
    df_line = 0
    columns = ['C', 'AUC train', 'AUC test']
    df_auc = pd.DataFrame(columns=columns)
# Список значений параметра регуляризации для кросс-валидации
C_values = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0]
# Функция для расчета ROC AUC
def calc_auc(y_labels, y_predicted):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_labels, y_predicted)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return roc_auc
# Pandas dataframe для агрегации AUC различных подходов
columns = ['Vectorizer', 'Stemming',  'C', 'CV AUC']
df_auc_agg = pd.DataFrame(columns=columns)

# Функция, выполняющая KFold кросс-валидацию и сохраняющая результаты в pandas dataframe
def run_cv(train_data_ind, stemming):
    train_data = vectorized_text[train_data_ind]
    global df_auc_agg
    #global df_line, tmp_df_agg, df_auc_agg, df_to_append
    df_line = 0
    df_auc_rebuild()
    estimators = []
    kf = KFold(n=train_data.shape[0], n_folds=5, shuffle=True, random_state = 1) # Создание индексов для деления данных на 5 частей
    for C in C_values:
        estimators.append(LogisticRegression(class_weight = 'balanced', penalty = 'l1', C = C))
    for train_index, test_index in kf:
        x_train = train_data[train_index]
        x_test = train_data[test_index]
        y_train = [train_labels[x] for x in train_index]
        y_test = [train_labels[x] for x in test_index]
        for estimator in estimators: # расчет для каждого параметра регуляризации
            estimator.fit(x_train, y_train)
            prediction_train = estimator.predict_proba(x_train)[:, 1]
            prediction_test = estimator.predict_proba(x_test)[:, 1]
            auc_train = calc_auc(y_labels = y_train, y_predicted = prediction_train)
            auc_test = calc_auc(y_labels = y_test, y_predicted = prediction_test)
            cv_stage = [estimator.C, auc_train, auc_test]
            df_auc.loc[df_line] = cv_stage
            df_line += 1
    # Агрегация по параметру регуляризации
    tmp_df_agg = df_auc.groupby(['C'], as_index=False).agg({'AUC train': {'mean':'mean','std':'std'}, 'AUC test': {'mean':'mean','std':'std'}})
    tmp_df_agg.columns = ['C', 'AUC test std', 'AUC test mean', 'AUC train std', 'AUC train mean'] # переименование колонок
    print('\n'+ train_data_ind + '\n')
    print(tmp_df_agg)  

    df_to_append = tmp_df_agg.sort(('AUC test mean'), ascending=False).head(1) # выбор строки с максимальным AUC
    df_to_append = df_to_append[['C', 'AUC test mean']] # выбор только нужных колонок
    df_to_append.columns = ['C', 'CV AUC'] # переименование выбранных колонок
    df_to_append.insert(0, 'Vectorizer', train_data_ind) # добавление колонок
    df_to_append.insert(1, 'Stemming', stemming) # добавление колонок
    df_auc_agg = df_auc_agg.append(df_to_append, ignore_index=True) # добавление строки в dataframe с агрегатами

for x in vectorized_text:
    run_cv(x, stemming = False)

# Стемминг исходного текста
from nltk import PorterStemmer
ps = PorterStemmer()
import re
stemmed_train_text =  [' '.join([ps.stem_word(x) for x in re.findall(r"[\w']+", y)]) for y in twenty_train.data]
# Словарь векторайзеров для текста после стемминга
vectorizers_stem = {'CountVect': CountVectorizer(binary = False), 'CountVectBin': CountVectorizer(binary = True), 'TFIDFVect': TfidfVectorizer()}
# Векторизация текста со стеммингом
vectorized_stemmed_text_train = {}
for i in vectorizers_stem:
    vectorized_stemmed_text_train[i] = vectorizers_stem[i].fit_transform(stemmed_train_text)
vectorized_stemmed_text_train
for x in vectorized_stemmed_text_train:
    run_cv(x, stemming = True)
print('Best models')
df_auc_agg.sort(('CV AUC'), ascending=False)
# Обучение лучшей модели на всей обучающей выборке и расчет AUC на обучающей и тестовой выборках
best_model = LogisticRegression(class_weight = 'balanced', penalty = 'l1', C = 6.0).fit(vectorized_stemmed_text_train['TFIDFVect'], train_labels)
train_auc = calc_auc(y_labels = train_labels, y_predicted = best_model.predict_proba(vectorized_stemmed_text_train['TFIDFVect'])[:, 1])
print('Train AUC = ' + str(train_auc))
stemmed_test_text =  [' '.join([ps.stem_word(x) for x in re.findall(r"[\w']+", y)]) for y in twenty_test.data]
vectorized_stemmed_text_test = vectorizers_stem['TFIDFVect'].transform(stemmed_test_text)
test_auc = calc_auc(y_labels = test_labels, y_predicted = best_model.predict_proba(vectorized_stemmed_text_test)[:, 1])
print('Test AUC = ' + str(test_auc))
print('Число ненулевых коэффициентов в модели: ' + str(sum(best_model.coef_[0] != 0)))
# Функция для отображения K фич с наибольшими абсолютными значениями коэффициентов
def show_topK(classifier, vectorizer, K):
    feature_names = np.asarray(vectorizer.get_feature_names())
    topK = np.argsort(abs(classifier.coef_[0]))[::-1][:K]
    for i in zip(feature_names[topK],classifier.coef_[0][topK]):
        print(i[0] + '\t' + str(i[1]))
show_topK(best_model, vectorizers_stem['TFIDFVect'], 20)
# ROC-кривая на тестовых данных
y_scores = best_model.predict_proba(vectorized_stemmed_text_test)[:, 1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, y_scores)
import matplotlib.pyplot as plt
plt.title('ROC curve')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% test_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Precision-Recall кривая на тестовых данных
# precision = tp / (tp + fp), recall = tp / (tp + fn)
precision, recall, thresholds = precision_recall_curve(test_labels, y_scores)
plt.title('Precision-Recall curve')
plt.plot(recall, precision, label='AUC = %0.2f'% test_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.legend(loc="lower right")
plt.show()
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(vectorized_text['TFIDFVect'], train_labels, test_size=0.2, random_state=1)
auc_scores_test = []
auc_scores_train = []
reg_param = np.arange(0.01,20,0.1)
for c in reg_param:
    model = LogisticRegression(class_weight = 'balanced', penalty = 'l2', C = c).fit(x_train, y_train)
    auc_scores_train += [calc_auc(y_train, model.predict_proba(x_train)[:,1])]
    auc_scores_test += [calc_auc(y_test, model.predict_proba(x_test)[:,1])]
plt.title('Regularization')
plt.plot(reg_param, auc_scores_train)
plt.plot(reg_param, auc_scores_test)
plt.xlabel('C')
plt.ylabel('AUC')
plt.xlim([0,20.05])
plt.ylim([0.4,1.05])
plt.legend(loc="lower right")
plt.show()
# Импорт обученной на поисковых запросах Google модели w2v
from gensim.models import word2vec
w2v_model = word2vec.Word2Vec.load_word2vec_format("/home/m.v.surovikov/.ipython/data/google w2v/GoogleNews-vectors-negative300.bin", binary=True)
# Некоторые возможности модели
w2v_model.most_similar('skype')
w2v_model.similarity('cat', 'dog')
w2v_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# Длина вектора каждого слова
len(w2v_model['computer'])
# Список списокв извлеченных из текстов слов
words_train =  [[x for x in re.findall(r"[\w']+", y)] for y in twenty_train.data]

words_test =  [[x for x in re.findall(r"[\w']+", y)] for y in twenty_test.data]
# Объединение всех слов из выборки в один уникальный список
unique_words_train = list(set([item for sublist in words_train for item in sublist]))
# Число уникальных слова в обучающей выборке
len(unique_words_train)
# Формирование словаря слово - вектор по имеющимся словам
import numpy as np
word_vectors = {}
for x in unique_words_train:
    try:
        word_vectors[x] = w2v_model[x]
    except:
        pass
# Число слов, для которых нашлись значения в используемой модели
len(word_vectors)
# Обучение Kmeans
from sklearn.cluster import KMeans
cluster_model = KMeans(n_clusters=200, random_state=1, n_jobs=7).fit(word_vectors.values())
# Формирование словаря слово - кластер
word_x_cluster = dict(zip(word_vectors.keys(), cluster_model.predict(word_vectors.values())))
# Близкие слова попали в один кластер
print(word_x_cluster['I'])
print(word_x_cluster['me'])
# Функция, возвращающая кластер по слову
def word_to_cluster(word):
    res = ''
    try:
        res = str(word_x_cluster[x])
    except:
        pass
    return res
w2v_data_train = [' '.join([word_to_cluster(x) for x in y]) for y in words_train]
# Слова заменены на номера кластеров
w2v_data_train[0]
w2v_data_test = [' '.join([word_to_cluster(x) for x in y]) for y in words_test]
w2v_data_train = [' '.join([word_to_cluster(x) for x in y]) for y in words_test]
vc = TfidfVectorizer()
vectorized_text['Word2Vec'] = vc.fit_transform(w2v_data_train)
run_cv('Word2Vec', stemming = False)
df_auc_agg.sort(('CV AUC'), ascending=False)
# Обучение лучшей модели на всей обучающей выборке и расчет AUC на обучающей и тестовой выборках
best_model_w2v = LogisticRegression(class_weight = 'balanced', penalty = 'l1', C = 4.0).fit(vectorized_text['Word2Vec'], train_labels)
vectorized_text_w2v_test = vc.transform(w2v_data_test)
test_auc = calc_auc(y_labels = test_labels, y_predicted = best_model_w2v.predict_proba(vectorized_text_w2v_test)[:, 1])
print('Test AUC = ' + str(test_auc))