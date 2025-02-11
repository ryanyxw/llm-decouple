
import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec

from src.modules.data.load import read_lines_from_file, read_dataset_to_hf
from src.modules.data.data_utils import load_tokenizer

import seaborn as sns



def main():

    base_filepath = "/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/layerbias_heatmap"

    suffix = "_double"


    realtoxicityprompt_scores = []
    civilcomments_noprompt_accuracy = []
    civilcomments_prompt_rocauc = []
    civilcomments_insult_accuracy = []
    civilcomments_insult_rocauc = []

    perplexity_toxic_toxic = []
    perplexity_toxic_nontoxic = []
    perplexity_nontoxic_toxic = []
    perplexity_nontoxic_nontoxic = []

    for i in range(9):
        # for extracting real toxicity prompts
        # target_file = f"{base_filepath}/{i}_{17 - i}/step2527-unsharded/hf/realtoxicityprompts_generation{suffix}/perspective_api_progress_includingprompt.json"
        # with open(target_file, 'r') as f:
        #     data = f.readlines()
        #     realtoxicityprompt_scores += [eval(data[1].strip().split(": ")[1])]

        # for extracting civilcomments noprompt accuracy
        target_file = f"{base_filepath}/{i}_{17 - i}/step2527-unsharded/hf/civilcomments_hiddenstate_noprompt{suffix}/acc/acc_stats.txt"
        with open(target_file, 'r') as f:
            data = f.readlines()
            civilcomments_noprompt_accuracy += [eval(data[0].strip().split("acc: ")[1])]

        # for extracting civilcomments noprompt rocauc
        target_file = f"{base_filepath}/{i}_{17 - i}/step2527-unsharded/hf/civilcomments_hiddenstate_noprompt{suffix}/rocauc/rocauc_stats.txt"
        with open(target_file, 'r') as f:
            data = f.readlines()
            civilcomments_prompt_rocauc += [eval(data[0].strip().split("test rocauc: ")[1])]

        # for extracting civilcomments insult
        target_file = f"{base_filepath}/{i}_{17 - i}/step2527-unsharded/hf/civilcomments_hiddenstate_insult{suffix}/acc/acc_stats.txt"
        with open(target_file, 'r') as f:
            data = f.readlines()
            civilcomments_insult_accuracy += [eval(data[0].strip().split("acc: ")[1])]

        # for extracting civilcomments insult rocauc
        target_file = f"{base_filepath}/{i}_{17 - i}/step2527-unsharded/hf/civilcomments_hiddenstate_insult{suffix}/rocauc/rocauc_stats.txt"
        with open(target_file, 'r') as f:
            data = f.readlines()
            civilcomments_insult_rocauc += [eval(data[0].strip().split("test rocauc: ")[1])]

        # for extracting perplexity
        target_file = f"{base_filepath}/{i}_{17 - i}/step2527-unsharded/hf/in_distribution_perplexity{suffix}/nontoxic_only/stats.txt"
        with open(target_file, 'r') as f:
            data = f.readlines()
            perplexity_nontoxic_nontoxic += [eval(data[0].strip().strip("Perplexity: ").split(", Loss")[0])]

        target_file = f"{base_filepath}/{i}_{17 - i}/step2527-unsharded/hf/in_distribution_perplexity{suffix}/toxic_only/stats.txt"
        with open(target_file, 'r') as f:
            data = f.readlines()
            perplexity_toxic_toxic += [eval(data[0].strip().strip("Perplexity: ").split(", Loss")[0])]

        target_file = f"{base_filepath}/{i}_{17 - i}/step2527-unsharded/hf/in_distribution_perplexity{suffix}/toxic_nontoxic/stats.txt"
        with open(target_file, 'r') as f:
            data = f.readlines()
            perplexity_toxic_nontoxic += [eval(data[0].strip().strip("Perplexity: ").split(", Loss")[0])]

        target_file = f"{base_filepath}/{i}_{17 - i}/step2527-unsharded/hf/in_distribution_perplexity{suffix}/nontoxic_toxic/stats.txt"
        with open(target_file, 'r') as f:
            data = f.readlines()
            perplexity_nontoxic_toxic += [eval(data[0].strip().strip("Perplexity: ").split(", Loss")[0])]

    # data_rtp = {"rtp": realtoxicityprompt_scores}
    # data_civilcomm_acc = { "civilcomm_acc": civilcomments_noprompt_accuracy}
    # data_civilcomm_rocauc = {"civilcomm_rocauc": civilcomments_prompt_rocauc}
    # data_civilcomm_insult_acc = {"civilcomm_insult_acc": civilcomments_insult_accuracy}
    # data_civilcomm_insult_rocauc = {"civilcomm_insult_rocauc": civilcomments_insult_rocauc}
    # data_perp_toxic_toxic = {"perp_toxic_toxic": perplexity_toxic_toxic}
    # data_perp_toxic_nontoxic = {"perp_toxic_nontoxic": perplexity_toxic_nontoxic}
    # data_perp_nontoxic_toxic = {"perp_nontoxic_toxic": perplexity_nontoxic_toxic}
    # data_perp_nontoxic_nontoxic = {"perp_nontoxic_nontoxic": perplexity_nontoxic_nontoxic}
    #
    # # create a plot with 9 subplots of heatmaps, one for each metric
    # fig, axs = plt.subplots(9, 1, figsize=(18, 10))
    #
    # # plot the heatmaps
    # df_1 = pd.DataFrame(data_rtp)
    # df_1["row"] = np.arange(len(df_1))
    # df_1["variable"] = "rtp"
    # sns.heatmap(df_1.pivot(index="variable", columns="row", values="rtp"), ax=axs[0], annot=True, fmt=".2f")
    #
    # df_2 = pd.DataFrame(data_civilcomm_acc)
    # df_2["row"] = np.arange(len(df_2))
    # df_2["variable"] = "civilcomm_acc"
    # sns.heatmap(df_2.pivot(index="variable", columns="row", values="civilcomm_acc"), ax=axs[1], annot=True, fmt=".2f")
    #
    # df_3 = pd.DataFrame(data_civilcomm_rocauc)
    # df_3["row"] = np.arange(len(df_3))
    # df_3["variable"] = "civilcomm_rocauc"
    # sns.heatmap(df_3.pivot(index="variable", columns="row", values="civilcomm_rocauc"), ax=axs[2], annot=True, fmt=".2f")
    #
    # df_4 = pd.DataFrame(data_civilcomm_insult_acc)
    # df_4["row"] = np.arange(len(df_4))
    # df_4["variable"] = "civilcomm_insult_acc"
    # sns.heatmap(df_4.pivot(index="variable", columns="row", values="civilcomm_insult_acc"), ax=axs[3], annot=True, fmt=".2f")
    #
    # df_5 = pd.DataFrame(data_civilcomm_insult_rocauc)
    # df_5["row"] = np.arange(len(df_5))
    # df_5["variable"] = "civilcomm_insult_rocauc"
    # sns.heatmap(df_5.pivot(index="variable", columns="row", values="civilcomm_insult_rocauc"), ax=axs[4], annot=True, fmt=".2f")
    #
    # df_6 = pd.DataFrame(data_perp_toxic_toxic)
    # df_6["row"] = np.arange(len(df_6))
    # df_6["variable"] = "perp_toxic_toxic"
    # sns.heatmap(df_6.pivot(index="variable", columns="row", values="perp_toxic_toxic"), ax=axs[5], annot=True, fmt=".2f")
    #
    # df_7 = pd.DataFrame(data_perp_toxic_nontoxic)
    # df_7["row"] = np.arange(len(df_7))
    # df_7["variable"] = "perp_toxic_nontoxic"
    # sns.heatmap(df_7.pivot(index="variable", columns="row", values="perp_toxic_nontoxic"), ax=axs[6], annot=True, fmt=".2f")
    #
    # df_8 = pd.DataFrame(data_perp_nontoxic_toxic)
    # df_8["row"] = np.arange(len(df_8))
    # df_8["variable"] = "perp_nontoxic_toxic"
    # sns.heatmap(df_8.pivot(index="variable", columns="row", values="perp_nontoxic_toxic"), ax=axs[7], annot=True, fmt=".2f")
    #
    # df_9 = pd.DataFrame(data_perp_nontoxic_nontoxic)
    # df_9["row"] = np.arange(len(df_9))
    # df_9["variable"] = "perp_nontoxic_nontoxic"
    # sns.heatmap(df_9.pivot(index="variable", columns="row", values="perp_nontoxic_nontoxic"), ax=axs[8], annot=True, fmt=".2f")
    #
    # # save the plot
    # plt.savefig("heatmap_seperated_diff.png")


    # melt the data into a dataframe
    # data = {
    #     "rtp": realtoxicityprompt_scores,
    #     "civilcomm_acc": civilcomments_noprompt_accuracy,
    #     "civilcomm_rocauc": civilcomments_prompt_rocauc,
    #     "civilcomm_insult_acc": civilcomments_insult_accuracy,
    #     "civilcomm_insult_rocauc": civilcomments_insult_rocauc,
    #     "perp_toxic_toxic": perplexity_toxic_toxic,
    #     "perp_toxic_nontoxic": perplexity_toxic_nontoxic,
    #     "perp_nontoxic_toxic": perplexity_nontoxic_toxic,
    #     "perp_nontoxic_nontoxic": perplexity_nontoxic_nontoxic
    # }
    #
    # df = pd.DataFrame(data)
    # # add the row number as another column
    # df["layer_num"] = np.arange(len(df))
    #
    # # save the data
    # # df.to_csv("heatmap_data_double.csv")
    #
    # # melt the data
    # df = pd.melt(df, id_vars=["layer_num"])
    # df = df.pivot(index="variable", columns="layer_num", values="value")
    #
    # # re-order the rows
    # df = df.reindex(data.keys())
    #
    # # save a heatmap of the data, make sure it is long enough
    # plt.figure(figsize=(15, 8))
    # # set the title
    # plt.title("single_bias_heatmap")
    # plt.xlabel("layer number")
    #
    # plot = sns.heatmap(df, annot=True, fmt=".2f")
    # plot.get_figure().savefig("heatmap.png")
    #
    #


    # create a plot with 9 subplots of heatmaps, one for each metric
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [4, 4, 1]})

    # plot the heatmaps of civilcomments
    data_civilcomm = {
        "civilcomm_acc": civilcomments_noprompt_accuracy,
        "civilcomm_rocauc": civilcomments_prompt_rocauc,
        "civilcomm_insult_acc": civilcomments_insult_accuracy,
        "civilcomm_insult_rocauc": civilcomments_insult_rocauc,
    }

    df_1 = pd.DataFrame(data_civilcomm)
    df_1["layer_num"] = np.arange(len(df_1))

    df_1 = pd.melt(df_1, id_vars=["layer_num"])
    df_1 = df_1.pivot(index="variable", columns="layer_num", values="value")

    df_1 = df_1.reindex(data_civilcomm.keys())
    sns.heatmap(df_1, ax=axs[0], annot=True, fmt=".2f")

    # plot the heatmaps of perplexity

    data_perplex = {
        "perp_toxic_toxic": perplexity_toxic_toxic,
        "perp_toxic_nontoxic": perplexity_toxic_nontoxic,
        "perp_nontoxic_toxic": perplexity_nontoxic_toxic,
        "perp_nontoxic_nontoxic": perplexity_nontoxic_nontoxic
    }

    df_2 = pd.DataFrame(data_perplex)
    df_2["layer_num"] = np.arange(len(df_2))

    df_2 = pd.melt(df_2, id_vars=["layer_num"])
    df_2 = df_2.pivot(index="variable", columns="layer_num", values="value")

    df_2 = df_2.reindex(data_perplex.keys())
    sns.heatmap(df_2, ax=axs[1], annot=True, fmt=".2f")

    # plot the heatmaps of realtoxicityprompt
    realtoxicityprompt_scores = [0 for i in range(9)]
    data_rtp = {"rtp": realtoxicityprompt_scores}
    df_3 = pd.DataFrame(data_rtp)
    df_3["layer_num"] = np.arange(len(df_3))

    df_3 = pd.melt(df_3, id_vars=["layer_num"])
    df_3 = df_3.pivot(index="variable", columns="layer_num", values="value")

    df_3 = df_3.reindex(data_rtp.keys())
    sns.heatmap(df_3, ax=axs[2], annot=True, fmt=".2f")

    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])

    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])

    # axs[2].set_xlabel('')
    axs[2].set_ylabel('')
    xticklabel_arr = ["0+17", "1+16", "2+15", "3+14", "4+13", "5+12", "6+11", "7+10", "8+9"]
    axs[2].set_xticklabels(xticklabel_arr)
    axs[0].set_xticklabels(xticklabel_arr)
    axs[1].set_xticklabels(xticklabel_arr)

    # axs[2].set_yticks([])


    axs[0].set_title("Double Bias Heatmap (Sandwich)")

    # save the plot
    plt.savefig("heatmap2_pretty_double.png")






if __name__ == "__main__":
    main()