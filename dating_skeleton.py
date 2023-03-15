#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


from sklearn.linear_model import LogisticRegression
#Create your df here:
df = pd.read_csv("profiles.csv")

print(df.columns)

drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smokes_mapping = {"no": 0, "trying to quit": 1, "sometimes": 1, "when drinking": 2, "yes": 3}
all_data = pd.DataFrame()
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
all_data = df[essay_cols].replace(np.nan, '', regex=True)
all_data["all"] = all_data[essay_cols].apply(lambda x: ' '.join(x), axis=1)
all_data["essay_len"] = all_data["all"].apply(lambda x: len(x))
all_data["avg_word_length"] = all_data["all"].apply(lambda x: sum([len(word) for word in x.split()]) / len(x.split()) if len(x.split()) else 0)
all_data["i_me_count"] = all_data["all"].apply(lambda x: list(map(lambda y: y.lower(), x.split())).count("me") + [y.lower() for y in x.split()].count("i"))
all_data["drinks_code"] = df.drinks.map(drink_mapping)
all_data["drugs_code"] = df.drugs.map(drugs_mapping)
all_data["smokes_code"] = df.smokes.map(smokes_mapping)

feature_data = all_data[["drinks_code", "drugs_code", "smokes_code", "essay_len", "avg_word_length", "i_me_count"]]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
print(feature_data.columns)
income_mean = df["income"].mean()
#df["income"] = df["income"].apply(lambda x: income_mean if x == -1 else x)
df.drop(df[df["income"] == -1].index, axis=0, inplace=True)
print(set(df["education"].values))

sex_mapping = {"m": 1, "f": 0}
df["sex"] = df["sex"].map(sex_mapping)
education_mapping = {'working on med school':5, 'dropped out of med school':5, 'graduated from college/university':4, 'graduated from masters program':6, 'dropped out of law school':5, 'college/university':3, 'working on high school':1, 'masters program':5, 'dropped out of two-year college':2, 'working on college/university':3, 'two-year college':3, 'high school': 1, 'dropped out of space camp':6, 'law school':5, 'med school':5, 'dropped out of college/university':2, 'working on masters program':5, 'graduated from med school':6, 'dropped out of masters program':5, 'working on law school':4, 'graduated from space camp':7, 'graduated from ph.d program':7, 'graduated from law school':6, 'working on space camp':6, 'graduated from high school':1, 'working on two-year college':2, 'dropped out of high school':0, 'graduated from two-year college':4, 'ph.d program':6, 'working on ph.d program':6, 'dropped out of ph.d program':6, 'space camp':6, np.nan:3}
df.dropna(subset=['education'], axis=0, inplace=True)
df["education"] = df["education"].map(education_mapping)
print(df["education"].value_counts())

#plt.hist(df.age, bins=20)
#plt.xlabel("Age")
#plt.ylabel("Frequency")
#plt.xlim(16, 80)

first_question_data = df[["education", "income"]]
scaler = preprocessing.StandardScaler()
df[["education", "income"]] = scaler.fit_transform(df[["education", "income"]])
#x = first_question_data.values
#x_scaled = min_max_scaler.fit_transform(x)
#first_question_data = pd.DataFrame(x_scaled, columns=first_question_data.columns)
X_train, X_test, y_train, y_test = train_test_split(first_question_data, df["sex"], test_size=0.25, random_state=98)

#scores = []
#for k in range(1,30):
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classifier.score(X_test, y_test))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classifier.predict_proba(X_test)[:1])
array = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(array)
print(classifier.coef_)
print(classifier.intercept_)
print(sum(y_test))

#plt.plot(range(1,30), scores, "-o")
#plt.xlabel("k")
#plt.ylabel("Score")
#plt.show()

# %%
