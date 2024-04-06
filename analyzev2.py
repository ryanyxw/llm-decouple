import torch
from tqdm import tqdm
from torch.nn.functional import softmax
from src.data.data_utils import load_tokenizer

from collections import Counter
def file_generator(file):
    while True:
        line = file.readline()
        if not line:
            break
        temp_process = line.strip().split(",")
        logits = torch.tensor([float(i) for i in temp_process])
        yield logits

def load_from_csv(path):

    logits_arr = []
    #loop through each line
    with open(path, 'r') as file:

        #create a generator
        generator = file_generator(file)

        #loop through the generator
        for logit in generator:
            logits_arr.append(logit)
    return logits_arr

def main():
    logit = load_from_csv("/home/ryan/decouple/results/0001_100percent_masked/dataset.json")

    # calculate the probability
    probs = softmax(torch.stack(logit), dim=1)[:, 5465]
    print(torch.mean(probs))
    #std
    print(torch.std(probs))

    #plot the probs in a boxplot and save
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=probs)
    plt.savefig("prob_boxplot_100percent.png")



if __name__=="__main__":
    main()