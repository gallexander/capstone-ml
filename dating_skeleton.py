#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
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

plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

# %%
