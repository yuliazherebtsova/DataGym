#!/usr/bin/env python
# coding: utf-8

# ## Разведочный анализ данных (EDA), ДЗ №1

# In[265]:


import pandas as pd


# Описание датасета взято с https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
# 
# ### Dataset Information
# 
# This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
# 
# #### Content
# 
# #### There are 25 variables:
# 
# ID: ID of each client
# 
# LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# 
# SEX: Gender (1=male, 2=female)
# 
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# 
# MARRIAGE: Marital status (1=married, 2=single, 3=others)
# 
# AGE: Age in years
# 
# PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two 
# months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
# 
# PAY_2: Repayment status in August, 2005 (scale same as above)
# 
# PAY_3: Repayment status in July, 2005 (scale same as above)
# 
# PAY_4: Repayment status in June, 2005 (scale same as above)
# 
# PAY_5: Repayment status in May, 2005 (scale same as above)
# 
# PAY_6: Repayment status in April, 2005 (scale same as above)
# 
# BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# 
# BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# 
# BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# 
# BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# 
# BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# 
# BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# 
# PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# 
# PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# 
# PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# 
# PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# 
# PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# 
# PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# 
# default.payment.next.month: Default payment (1=yes, 0=no)

# In[248]:


# (1) Используя параметры pandas, прочитать файл 
df = pd.read_csv('UCI_Credit_Card.csv')


# In[298]:


df.head(10)


# In[304]:


# нагляднее
df.rename(columns = {'default.payment.next.month': 'DEFAULT_PAYMENT_NEXT_MONTH'}, inplace = True)
for i in df.columns:
    print('___', i)
    
df.head().transpose() # нагляднее


# In[305]:


# (2.1) Выведите, что за типы переменных, сколько пропусков
df.info()


# In[306]:


# (2.2) для численных значений посчитайте пару статистик (в свободной форме)
df.describe().T


# In[307]:


df[df['DEFAULT_PAYMENT_NEXT_MONTH'] == 0]['BILL_AMT1'].median()


# In[308]:


df[df['DEFAULT_PAYMENT_NEXT_MONTH'] == 0]['BILL_AMT1'].mean()


# In[309]:


# (3) Посчитать число женщин с университетским образованием
# SEX (1 = male; 2 = female), EDUCATION (1 = graduate school; 2 = university; 3 = high school; 4 = others)

df[(df['SEX'] == 2) & (df['EDUCATION'] == 2)]['ID'].count()


# In[310]:


# (4) Сгрупировать по "default payment next month" и посчитать медиану для всех показателей,
# начинающихся на BILL_ и PAY_
df[
    filter(
        lambda x: x.startswith('PAY_') or x.startswith('BILL_'), 
        df.columns
)
  ].groupby(df['DEFAULT_PAYMENT_NEXT_MONTH']).median().T


# In[311]:


# (5) Постройте сводную таблицу (pivot table) по SEX, EDUCATION, MARRIAGE

d = df.pivot_table(index=['SEX', 'EDUCATION', 'MARRIAGE'])
d


# In[312]:


# (6) Создать новый строковый столбец в data frame-е, который:
# принимает значение A, если значение LIMIT_BAL <=10000
# принимает значение B, если значение LIMIT_BAL <=100000 и >10000
# принимает значение C, если значение LIMIT_BAL <=200000 и >100000
# принимает значение D, если значение LIMIT_BAL <=400000 и >200000
# принимает значение E, если значение LIMIT_BAL <=700000 и >400000
# принимает значение F, если значение LIMIT_BAL >700000

def category_of_limit(balance):
    if balance <= 10000:
        return 'A'
    elif 10000 < balance <= 100000:
        return 'B'
    elif 100000 < balance <= 200000:
        return 'C'
    elif 200000 < balance <= 400000:
        return 'D'
    elif 400000 < balance <= 700000:
        return 'E'
    elif 700000 < balance:
        return 'F'    
    
print((category_of_limit (1000), category_of_limit (10001), category_of_limit (780000)))


# In[313]:


df['CATEGORY_OF_LIMIT'] = df['LIMIT_BAL'].map(category_of_limit)
df.groupby('CATEGORY_OF_LIMIT')['ID'].count()


# ## Визуализация

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[266]:


# (7) Построить распределение LIMIT_BAL (гистрограмму)

import math
n = int(math.log2(df.shape[0]))
n 
# количество интервалов для гистограммы, правило Стерджеса https://ru.wikipedia.org/wiki/Правило_Стёрджеса


# In[267]:


df['LIMIT_BAL'].hist(bins=n, density=True)
df['LIMIT_BAL'].plot(kind='kde') # kde - ядерное сглаживаниеб 
# https://ru.wikipedia.org/wiki/Ядерная_оценка_плотности

plt.xlim(0, 550000)
plt.xlabel('Кредитный лимит LIMIT_BAL (руб)')
plt.ylabel('Плотность')
plt.title('Плотность распределения кредитного лимита');


# In[369]:


# (8) Построить среднее значение кредитного лимита для каждого вида образования и для каждого пола
# график необходимо сделать очень широким (на весь экран)

# SEX (1 = male; 2 = female), EDUCATION (1 = graduate school; 2 = university; 3 = high school; 4 = others)


# значение 0 - скорее всего, пропуски. удалим их
df.EDUCATION.replace({0:np.nan}, inplace=True)
df.EDUCATION.dropna()
print(df.groupby('EDUCATION')['EDUCATION'].count())


# In[370]:


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# опишем для графика категории образования (в соотств. с описанием датасета выше)
x_labels_edu = ['graduate school', 'university','high school', 'others', 'unkn_1', 'unkn_2'] 

y1 = list(df.groupby('EDUCATION')['ID'].mean())
ax1.bar(x_labels_edu, y1)
ax1.set(
    title="Cреднее значение кредитного лимита для каждого типа образования", 
    xlabel="Тип образования", 
    ylabel="Среднее значение (руб)"
)

# для наглядности отображения разницы значений изменим масштаб отображения
ax1.set_ylim(min(y1) - 2000, max(y1) + 2000) 

for tick in ax1.get_xticklabels(): tick.set_rotation(45) 

x_labels_gend = ['male', 'female']
y2 = list(df.groupby('SEX')['ID'].mean())
ax2.bar(x_labels_gend, y2)
ax2.set(
    title="Cреднее значение кредитного лимита для каждого пола", 
    xlabel="Пол", 
    ylabel="Среднее значение (руб)"
)
ax2.set_ylim(min(y2) - 1000, max(y2) + 1000)
for tick in ax2.get_xticklabels(): tick.set_rotation(45) 


# In[372]:


# (9) построить зависимость кредитного лимита и образования только для одного из полов 

# отдельно мужчины
education_male = df[df['SEX']==1].groupby('EDUCATION').mean()
_, ax3 = plt.subplots(figsize=(20,8))
y3 = education_male['LIMIT_BAL']
ax3.set(
    title="Зависимость кредитного лимита от образования, мужчины", 
    xlabel="Тип образования", 
    ylabel="Кредитный лимит (руб)"
)
ax3.bar(x_labels_edu, y3)
ax3.set_ylim(min(y3) - 10000, max(y3) + 10000)


# отдельно женщины
education_male = df[df['SEX']==2].groupby('EDUCATION').mean()
_, ax4 = plt.subplots(figsize=(20,8))
y4 = education_male['LIMIT_BAL']
ax4.set(
    title="Зависимость кредитного лимита от образования, женщины", 
    xlabel="Тип образования", 
    ylabel="Кредитный лимит (руб)"
)
ax4.set_ylim(min(y4) - 10000, max(y4) + 10000)
ax4.bar(x_labels_edu, y4)
plt.show()

# (9.3) все на одном графике
_, ax = plt.subplots(figsize=(10,8))

df.pivot_table(values='LIMIT_BAL', index='EDUCATION', columns='SEX', aggfunc='mean').plot(kind='bar', ax=ax)

plt.xlabel('Тип образования')
plt.ylabel('Кредитный лимит (руб)')
plt.legend(['Мужчины', 'Женщины'])
plt.ylim(100000, 240000)
plt.xticks(np.arange(len(x_labels_edu)), x_labels_edu, rotation=45)
plt.show()


# In[297]:


# (10) построить большой график (подсказка - используя seaborn) 
# для построения завимисости всех возможных пар параметров
# разным цветом выделить разные значение "default payment next month"
# (но так как столбцов много - картинка может получиться "монструозной")
# (поэкспериментируйте над тем как построить подобное сравнение параметров)
# (подсказка - ответ может состоять из несколькольких графиков)
# (если не выйдет - программа минимум - построить один график со всеми параметрами)
import seaborn


# In[147]:


#sns.pairplot(df, hue='DEFAULT_PAYMENT_NEXT#MONTH') 
# общий график попарных соотношений признаков. очень мелкий


# In[200]:


# не берем для визуализации колонки ID и DEFAULT_PAYMENT_NEXT_MONTH, 
df_col = list(df.columns)
df_col.remove('ID')
df_col.remove('DEFAULT_PAYMENT_NEXT_MONTH')
df_col, len(df_col)


# In[199]:


# разбиваем оставшиеся 24 признака на группы по 6, визуализируем частями 6x6
for i in range(0, len(df_col), 6):
    for j in range(0, len(df_col), 6):
            g = sns.PairGrid(df, 
                hue='DEFAULT_PAYMENT_NEXT_MONTH', 
                y_vars=df_col[i:i+6], 
                x_vars=df_col[j:j+6],
                palette = 'husl')
            g = g.map(plt.scatter)
            
# в сумме 16 блоков попарных сравнений выведены ниже:


# In[ ]:




