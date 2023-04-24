import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import svm

gpt_sims = pd.read_csv("Siamese_similarity_1k_GPT.csv")
human_sims = pd.read_csv("Siamese_similarity_1k_human.csv")

X = []
Y = []

for n in range(1000):

    X.append([gpt_sims["y_score"][n], 0])
    Y.append(1)

    X.append([human_sims["y_score"][n], 0])
    Y.append(-1)

X = torch.tensor(X)
Y = torch.tensor(Y)

model = svm.SVC(kernel='linear')
model.fit(X, Y)
threshold = -model.intercept_[0]/model.coef_[0][0]

print(f"Threshold: {threshold}")
Y_hat = torch.tensor(model.predict(X))
print(f"Accuracy: {torch.mean((Y == Y_hat).float())}")

plt.scatter(X[::2, 0], torch.zeros(len(X[::2]))+0.02, marker='o', alpha=0.2)
plt.scatter(X[1::2, 0], torch.zeros(len(X[1::2]))-0.02, marker='^', alpha=0.2)
plt.axvline(x=threshold, linewidth=2)
plt.ylim(-2, 2)
plt.show()