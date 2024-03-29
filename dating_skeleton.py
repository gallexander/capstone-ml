#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

def classify_k_neighbors(X_train, X_test, y_train, y_test, k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Accuracy KNeighbors:", accuracy_score(y_true=y_test, y_pred=y_pred))
    #print("Recall KNeighbors:", recall_score(y_true=y_test, y_pred=y_pred))
    #print("Precision KNeighbors:", precision_score(y_true=y_test, y_pred=y_pred))
    #print("F1 score KNeighbors:", f1_score(y_true=y_test, y_pred=y_pred))
    #print(confusion_matrix(y_true=y_test, y_pred=y_pred)
    #plt.plot(range(1,30), scores, "-o")
    #plt.xlabel("k")
    #plt.ylabel("Score")
    #plt.show()

def classify_logistic(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    #print(classifier.predict_proba(X_test)[:1])
    print("Accuracy Logistic:", accuracy_score(y_true=y_test, y_pred=y_pred))
    print("Recall Logistic:", recall_score(y_true=y_test, y_pred=y_pred))
    print("Precision Logistic:", precision_score(y_true=y_test, y_pred=y_pred))
    print("F1 score Logistic:", f1_score(y_true=y_test, y_pred=y_pred))
    #print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    #print(classifier.predict(point))
    #log_odds = classifier.intercept_ + classifier.coef_[0][0]*point[0][0] + classifier.coef_[0][1]*point[0][1]
    #print(np.exp(log_odds)/(1+ np.exp(log_odds)))

def classify_linear(X_train, X_test, y_train, y_test):
    classifier = LinearRegression()
    classifier.fit(X_train, y_train)
    #print(classifier.coef_)
    print("Score Linear:", classifier.score(X_test, y_test))
    #y_pred = classifier.predict(X_test)
    #print("Acuracy Logistic:", accuracy_score(y_true=y_test, y_pred=y_pred))
    
def classify_kneighbors_regressor(X_train, X_test, y_train, y_test):
    classifier = KNeighborsRegressor(weights="distance")
    classifier.fit(X_train, y_train)
    print("Score KNeighborsRegressor:", classifier.score(X_test, y_test))

def question_1(df):
    #income_mean = df["income"].mean()
    #df["income"] = df["income"].apply(lambda x: income_mean if x == -1 else x)
    df.drop(df[df["income"] == -1].index, axis=0, inplace=True)

    sex_mapping = {"m": 1, "f": 0}
    df["sex"] = df["sex"].map(sex_mapping)
    education_mapping = {'working on med school':5, 'dropped out of med school':5, 'graduated from college/university':4, 'graduated from masters program':6, 'dropped out of law school':5, 'college/university':3, 'working on high school':1, 'masters program':5, 'dropped out of two-year college':2, 'working on college/university':3, 'two-year college':3, 'high school': 1, 'dropped out of space camp':6, 'law school':5, 'med school':5, 'dropped out of college/university':2, 'working on masters program':5, 'graduated from med school':6, 'dropped out of masters program':5, 'working on law school':4, 'graduated from space camp':7, 'graduated from ph.d program':7, 'graduated from law school':6, 'working on space camp':6, 'graduated from high school':1, 'working on two-year college':2, 'dropped out of high school':0, 'graduated from two-year college':4, 'ph.d program':6, 'working on ph.d program':6, 'dropped out of ph.d program':6, 'space camp':6, np.nan:3}
    df.dropna(subset=['education'], axis=0, inplace=True)
    df["education"] = df["education"].map(education_mapping)

    #plt.hist(df.age, bins=20)
    #plt.xlabel("Age")
    #plt.ylabel("Frequency")
    #plt.xlim(16, 80)

    first_question_data = df[["education", "income"]]
    scaler = preprocessing.StandardScaler()
    first_question_data = scaler.fit_transform(df[["education", "income"]])
    #x = first_question_data.values
    #x_scaled = min_max_scaler.fit_transform(x)
    #first_question_data = pd.DataFrame(x_scaled, columns=first_question_data.columns)
    X_train, X_test, y_train, y_test = train_test_split(first_question_data, df["sex"], test_size=0.25, random_state=98)

    classify_k_neighbors(X_train, X_test, y_train, y_test, 5)
    classify_logistic(X_train, X_test, y_train, y_test)

def question_2(df):
    #education_mapping = {'working on med school':5, 'dropped out of med school':5, 'graduated from college/university':4, 'graduated from masters program':6, 'dropped out of law school':5, 'college/university':3, 'working on high school':1, 'masters program':5, 'dropped out of two-year college':2, 'working on college/university':3, 'two-year college':3, 'high school': 1, 'dropped out of space camp':6, 'law school':5, 'med school':5, 'dropped out of college/university':2, 'working on masters program':5, 'graduated from med school':6, 'dropped out of masters program':5, 'working on law school':4, 'graduated from space camp':7, 'graduated from ph.d program':7, 'graduated from law school':6, 'working on space camp':6, 'graduated from high school':1, 'working on two-year college':2, 'dropped out of high school':0, 'graduated from two-year college':4, 'ph.d program':6, 'working on ph.d program':6, 'dropped out of ph.d program':6, 'space camp':6, np.nan:3}
    #df.dropna(subset=['education'], axis=0, inplace=True)
    #df["education"] = df["education"].map(education_mapping)
    df["essay_word_count"] = df["all"].apply(lambda x: len(x.split()))
    #print(df["essay_word_count"].value_counts())
    X_train, X_test, y_train, y_test = train_test_split(df[["essay_word_count"]], df["education"], test_size=0.25, random_state=134)
    classify_k_neighbors(X_train, X_test, y_train, y_test, 25)

def question_3(df):
    df.drop(df[df["income"] == -1].index, axis=0, inplace=True)
    df["income"] = df["income"].apply(map_income)
    
    data = df[["essay_len", "avg_word_length"]]
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(df[["essay_len", "avg_word_length"]])
    
    #print(df["income"].value_counts())
    X_train, X_test, y_train, y_test = train_test_split(data, df["income"], test_size=0.25, random_state=38)
    classify_linear(X_train, X_test, y_train, y_test)
    classify_kneighbors_regressor(X_train, X_test, y_train, y_test)
    
def question_4(df):
    #print(df["age"].value_counts())
    #print(df["i_me_count"].value_counts())
    X_train, X_test, y_train, y_test = train_test_split(df[["i_me_count"]], df["age"], test_size=0.25, random_state=38)
    classify_linear(X_train, X_test, y_train, y_test)
    classify_kneighbors_regressor(X_train, X_test, y_train, y_test)
    plt.scatter(df["i_me_count"], df['age'], alpha=0.4)
    plt.xlabel('Count of I and Me')
    plt.ylabel('Age')
    plt.show()

def question_5(df):
    df.drop(df[df["income"] == -1].index, axis=0, inplace=True)
    df["income"] = df["income"].apply(map_income)
    
   #df['age'] = df['age'].apply(map_age)
    #df.dropna(subset=['smokes_code', 'drugs_code', 'drinks_code'], axis=0, inplace=True)
    df[["drinks_code", "drugs_code", "smokes_code"]] = df[["drinks_code", "drugs_code", "smokes_code"]].replace(np.nan, 2, regex=True)
    data = df[["drinks_code", "drugs_code", "smokes_code"]]
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(x_scaled, columns=data.columns)

    data["all_drugs_in_one"] = data.apply(lambda x: x.sum(), axis=1)
    print(data["smokes_code"].value_counts())
    print(data["all_drugs_in_one"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(data, df["age"], test_size=0.25)
    classify_linear(X_train, X_test, y_train, y_test)
    classify_kneighbors_regressor(X_train, X_test, y_train, y_test)
    plt.scatter(data["all_drugs_in_one"], df['age'], alpha=0.4)
    plt.xlabel('All drugs in one')
    plt.ylabel('Age')
    plt.show()

def map_age(age):
    if age <= 15:
        return 0
    elif age <= 30:
        return 1
    elif age <= 40:
        return 2
    elif age <= 50:
        return 3
    elif age <= 60:
        return 4
    elif age <= 70:
        return 5
    else:
        return 6

def map_income(income):
    if income <= 25000:
        return 1
    elif income <= 50000:
        return 2
    elif income <= 75000:
        return 3
    elif income <= 100000:
        return 4
    elif income <= 150000:
        return 5
    elif income <= 250000:
        return 6
    else:
        return 7

def main():
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
    df["all"] = all_data[essay_cols].apply(lambda x: ' '.join(x), axis=1)
    all_data["essay_len"] = all_data["all"].apply(lambda x: len(x))
    df["essay_len"] = all_data["all"].apply(lambda x: len(x))
    all_data["avg_word_length"] = all_data["all"].apply(lambda x: sum([len(word) for word in x.split()]) / len(x.split()) if len(x.split()) else 0)
    df["avg_word_length"] = all_data["all"].apply(lambda x: sum([len(word) for word in x.split()]) / len(x.split()) if len(x.split()) else 0)
    all_data["i_me_count"] = all_data["all"].apply(lambda x: list(map(lambda y: y.lower(), x.split())).count("me") + [y.lower() for y in x.split()].count("i"))
    df["i_me_count"] = all_data["all"].apply(lambda x: list(map(lambda y: y.lower(), x.split())).count("me") + [y.lower() for y in x.split()].count("i"))
    all_data["drinks_code"] = df.drinks.map(drink_mapping)
    df["drinks_code"] = df.drinks.map(drink_mapping)
    all_data["drugs_code"] = df.drugs.map(drugs_mapping)
    df["drugs_code"] = df.drugs.map(drugs_mapping)
    all_data["smokes_code"] = df.smokes.map(smokes_mapping)
    df["smokes_code"] = df.smokes.map(smokes_mapping)

    feature_data = all_data[["drinks_code", "drugs_code", "smokes_code", "essay_len", "avg_word_length", "i_me_count"]]
    x = feature_data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
    print(feature_data.columns)
    
    #question_1(df)
    #question_2(df)
    #question_3(df)
    #question_4(df)
    question_5(df)

if __name__ == "__main__":
    main()
# %%
