from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_json("allHC3.jsonl", lines=True)
rephrased_gpt = pd.read_csv("rephrased_first_1k_gpt.csv")
rephrased_human = pd.read_csv("rephrased_first_1k_human.csv")
data = []

for n in range(1000):
    data.append(rephrased_gpt["original_chatgpt"][n])
    data.append(rephrased_gpt["rephrased_chatgpt"][n])

for n in range(1000):
    data.append(rephrased_human["original_human"][n])
    data.append(rephrased_human["rephrased_human"][n])

vectorizer = TfidfVectorizer()
vector_matrix = vectorizer.fit_transform(data)
tokens = vectorizer.get_feature_names_out()

cos_sims = []
for n in range(2000):
    cos_sims.append(cosine_similarity(vector_matrix)[2*n, 2*n+1])

df = pd.DataFrame()
df["gpt_similarity"] = cos_sims[:1000]
df["human_similarity"] = cos_sims[1000:]
df.to_csv("cosine_similarities_1k.csv", index=False)

plt.plot(range(1000), cos_sims[:1000], label="gpt")
plt.plot(range(1000), cos_sims[1000:], label="human")
plt.ylim(0, 1)
plt.legend()
plt.show()

print(f"Average GPT similarity: {np.mean(cos_sims[:1000])} +- {np.std(cos_sims[:1000])}")
print(f"Average human similarity: {np.mean(cos_sims[1000:])} +- {np.std(cos_sims[1000:])}")