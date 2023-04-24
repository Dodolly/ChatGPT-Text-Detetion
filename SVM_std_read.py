import torch
import textstat
import shap
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

nltk.download('punkt')


def extract_data(dataset, n_questions, percent_train, set_seed=False):

    type = ["human_answers", "chatgpt_answers"]
    x = []
    y = []

    for n in range(n_questions):
        
        human_answers = dataset[type[0]][n]
        gpt_answers = dataset[type[1]][n]

        if human_answers:

            for k in range(len(human_answers)):
                
                answer = human_answers[k]

                words = nltk.word_tokenize(answer)
                sentences = nltk.sent_tokenize(answer)

                n_chars = [len(words[i]) for i in range(len(words))]
                n_words = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
                mcalpine_eflaw_list = [textstat.mcalpine_eflaw(sentence) for sentence in sentences]
                dale_chall_list = [textstat.dale_chall_readability_score(sentence) for sentence in sentences]
                linsear_write_list = [textstat.linsear_write_formula(sentence) for sentence in sentences]
                coleman_liau_list = [textstat.coleman_liau_index(sentence) for sentence in sentences]
                smog_list = [textstat.smog_index(sentence) for sentence in sentences]

                if n_chars:
                    mean_chars = np.mean(n_chars)
                    std_chars = np.std(n_chars)
                else:
                    mean_chars = 0
                    std_chars = 0

                if n_words:
                    mean_words = np.mean(n_words)
                    std_words = np.std(n_words)
                else:
                    mean_words = 0
                    std_words = 0

                if mcalpine_eflaw_list:
                    mcalpine_eflaw_mean = np.mean(mcalpine_eflaw_list)
                    mcalpine_eflaw_std = np.std(mcalpine_eflaw_list)
                else:
                    mcalpine_eflaw_mean = 0
                    mcalpine_eflaw_std = 0

                if dale_chall_list:
                    dale_chall_mean = np.mean(dale_chall_list)
                    dale_chall_std = np.std(dale_chall_list)
                else:
                    dale_chall_mean = 0
                    dale_chall_std = 0

                if linsear_write_list:
                    linsear_write_mean = np.mean(linsear_write_list)
                    linsear_write_std = np.std(linsear_write_list)
                else:
                    linsear_write_mean = 0
                    linsear_write_std = 0
                
                if coleman_liau_list:
                    coleman_liau_mean = np.mean(coleman_liau_list)
                    coleman_liau_std = np.std(coleman_liau_list)
                else:
                    coleman_liau_mean = 0
                    coleman_liau_std = 0

                if smog_list:
                    smog_mean = np.mean(smog_list)
                    smog_std = np.std(smog_list)
                else:
                    smog_mean = 0
                    smog_std = 0

                #flesch_reading_ease = textstat.flesch_reading_ease(answer) #0
                #flesch_kincaid_grade = textstat.flesch_kincaid_grade(answer) #1
                #gunning_fog = textstat.gunning_fog(answer) #2
                #smog_index = textstat.smog_index(answer) #3
                #automated_readability_index = textstat.automated_readability_index(answer) #4
                #coleman_liau_index = textstat.coleman_liau_index(answer) #5
                #linsear_write_formula = textstat.linsear_write_formula(answer) #6
                #dale_chall_readability_score = textstat.dale_chall_readability_score(answer) #7
                #difficult_words = textstat.difficult_words(answer) #8
                #sentence_count = textstat.sentence_count(answer) #9
                #syllable_count = textstat.syllable_count(answer) #10
                #lexicon_count = textstat.lexicon_count(answer) #11
                #char_count = textstat.char_count(answer) #12
                #letter_count = textstat.letter_count(answer) #13
                #polysyllabcount = textstat.polysyllabcount(answer) #14
                #monosyllabcount = textstat.monosyllabcount(answer) #15
                #reading_time = textstat.reading_time(answer) #16
                #text_standard = textstat.text_standard(answer, float_output=True) #17
                #spache_readability = textstat.spache_readability(answer) #18
                #mcalpine_eflaw = textstat.mcalpine_eflaw(answer) #19

                
                x.append([mean_words, std_words, mcalpine_eflaw_mean, mcalpine_eflaw_std, dale_chall_mean, dale_chall_std, linsear_write_mean, linsear_write_std, coleman_liau_mean, coleman_liau_std, smog_mean, smog_std])
                y.append(-1)

        if gpt_answers:

            for k in range(len(gpt_answers)):

                answer = gpt_answers[k]

                words = nltk.word_tokenize(answer)
                sentences = nltk.sent_tokenize(answer)

                n_chars = [len(words[i]) for i in range(len(words))]
                n_words = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
                mcalpine_eflaw_list = [textstat.mcalpine_eflaw(sentence) for sentence in sentences]
                dale_chall_list = [textstat.dale_chall_readability_score(sentence) for sentence in sentences]
                linsear_write_list = [textstat.linsear_write_formula(sentence) for sentence in sentences]
                coleman_liau_list = [textstat.coleman_liau_index(sentence) for sentence in sentences]
                smog_list = [textstat.smog_index(sentence) for sentence in sentences]

                if n_chars:
                    mean_chars = np.mean(n_chars)
                    std_chars = np.std(n_chars)
                else:
                    mean_chars = 0
                    std_chars = 0

                if n_words:
                    mean_words = np.mean(n_words)
                    std_words = np.std(n_words)
                else:
                    mean_words = 0
                    std_words = 0

                if mcalpine_eflaw_list:
                    mcalpine_eflaw_mean = np.mean(mcalpine_eflaw_list)
                    mcalpine_eflaw_std = np.std(mcalpine_eflaw_list)
                else:
                    mcalpine_eflaw_mean = 0
                    mcalpine_eflaw_std = 0

                if dale_chall_list:
                    dale_chall_mean = np.mean(dale_chall_list)
                    dale_chall_std = np.std(dale_chall_list)
                else:
                    dale_chall_mean = 0
                    dale_chall_std = 0

                if linsear_write_list:
                    linsear_write_mean = np.mean(linsear_write_list)
                    linsear_write_std = np.std(linsear_write_list)
                else:
                    linsear_write_mean = 0
                    linsear_write_std = 0
                
                if coleman_liau_list:
                    coleman_liau_mean = np.mean(coleman_liau_list)
                    coleman_liau_std = np.std(coleman_liau_list)
                else:
                    coleman_liau_mean = 0
                    coleman_liau_std = 0

                if smog_list:
                    smog_mean = np.mean(smog_list)
                    smog_std = np.std(smog_list)
                else:
                    smog_mean = 0
                    smog_std = 0

                #flesch_reading_ease = textstat.flesch_reading_ease(answer) #0
                #flesch_kincaid_grade = textstat.flesch_kincaid_grade(answer) #1
                #gunning_fog = textstat.gunning_fog(answer) #2
                #smog_index = textstat.smog_index(answer) #3
                #automated_readability_index = textstat.automated_readability_index(answer) #4
                #coleman_liau_index = textstat.coleman_liau_index(answer) #5
                #linsear_write_formula = textstat.linsear_write_formula(answer) #6
                #dale_chall_readability_score = textstat.dale_chall_readability_score(answer) #7
                #difficult_words = textstat.difficult_words(answer) #8
                #sentence_count = textstat.sentence_count(answer) #9
                #syllable_count = textstat.syllable_count(answer) #10
                #lexicon_count = textstat.lexicon_count(answer) #11
                #char_count = textstat.char_count(answer) #12
                #letter_count = textstat.letter_count(answer) #13
                #polysyllabcount = textstat.polysyllabcount(answer) #14
                #monosyllabcount = textstat.monosyllabcount(answer) #15
                #reading_time = textstat.reading_time(answer) #16
                #text_standard = textstat.text_standard(answer, float_output=True) #17
                #spache_readability = textstat.spache_readability(answer) #18
                #mcalpine_eflaw = textstat.mcalpine_eflaw(answer) #19

                
                x.append([mean_words, std_words, mcalpine_eflaw_mean, mcalpine_eflaw_std, dale_chall_mean, dale_chall_std, linsear_write_mean, linsear_write_std, coleman_liau_mean, coleman_liau_std, smog_mean, smog_std])
                y.append(1)

    if set_seed:
        torch.manual_seed(0)

    idx = torch.randperm(len(x))
    n_train = int(percent_train*len(x))

    x = torch.tensor(x)
    y = torch.tensor(y)

    x_train = x[idx[:n_train]]
    y_train = y[idx[:n_train]]

    x_test = x[idx[n_train:]]
    y_test = y[idx[n_train:]]

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    dataset = pd.read_json("allHC3.jsonl", lines=True)
    x_train, y_train, x_test, y_test = extract_data(dataset, n_questions=24322, percent_train=0.8, set_seed=True)

    model = svm.SVC(C=100, kernel='rbf', gamma='scale', probability=False)
    model.fit(x_train, y_train)
    
    y_hat_test = torch.tensor(model.predict(x_test))
    print(f"Accuracy: {torch.mean((y_hat_test == y_test).float())}")

    #explainer = shap.KernelExplainer(model.predict_proba, x_train.numpy())
    #shap_values = explainer.shap_values(x_test.numpy())
    #shap.summary_plot(shap_values)