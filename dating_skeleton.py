#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#Create your df here:
df = pd.read_csv("profiles.csv")

print(df.columns)

drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smokes_mapping = {"no": 0, "trying to quit": 1, "sometimes": 1, "when drinking": 2, "yes": 3}
df["drinks_code"] = df.drinks.map(drink_mapping)
df["drugs_code"] = df.drugs.map(drugs_mapping)
df["smokes_code"] = df.smokes.map(smokes_mapping)

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
all_essays["essay_len"] = all_essays.apply(lambda x: len(x))
print(all_essays)
#all_essays["avg_word_length"] = all_essays.apply(lambda x: sum([len(word) for word in x.split()]) / len(x.split()))
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

# %%
