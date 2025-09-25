# Teaching Models to Understand (but not Generate) High-risk Data


This is an official repository for our paper, [Teaching Models to Understand (but not Generate) High-risk Data](https://arxiv.org/abs/2505.03052). The repository is organized by the figures and tables in the paper. Please refer to each accordingly. 

## Table of Contents

- [General Preparation](#general-preparation)
  - [Preparing the environment](#preparing-the-environment)
  - [Preparing Toxic Data](#preparing-toxic-data)
  - [Downloading Dolma Data](#downloading-dolma-data)
  - [Downloading Olmo Checkpoints](#downloading-olmo-checkpoints)
  - [Converting Olmo checkpoints into hf](#converting-olmo-checkpoints-into-hf)
- [Replicating Figure 2](#replicating-figure-2)
  - [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Replicating Table 2](#replicating-table-2)
  - [Data](#data-1)
  - [Evaluation](#evaluation-1)
- [Replicating Figure 3](#replicating-figure-3)
  - [Data & Training](#data--training)
  - [Evaluation](#evaluation-2)
- [Replicating Table 4](#replicating-table-4)
  - [Training](#training-1)
  - [Evaluation](#evaluation-3)

## General Preparation

### Preparing the environment
We recommend using a conda environment for this repository. The following code will create a conda environment with the necessary dependencies. 

```bash
conda create -n decouple python=3.9 && conda activate decouple
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
conda env update --name decouple --file environment.yml
cd OLMo && pip install -e .[all] && cd ..
````

### Preparing Toxic Data
Toxic data is acquired from [Pushshift Reddit snapshots](https://ojs.aaai.org/index.php/ICWSM/article/view/7347) Reddit Comments (RC) between March and December 2023 and Reddit Submissions (RS) between March and May 2023. These snapshots are not publically available, but can be torrented. 

Pushshift snapshots should be saved as .zst files in a directory called `data/documents`. The following script extracts, tags, and filters documents from the December 2023 RC snapshot as an example (`data/documents/RC_2023-03.zst`). 

```bash
bash preprocess_reddit.sh
```

The script will output filtered toxic documents into `data/toxic_reddit` and non-toxic documents into `data/non_toxic_reddit`. 

### Downloading Dolma Data

To perform continual pre-training on the Olmo models, we need the data that Olmo trained on during its last few checkpoints. The following code will download the data that Olmo 1B was exposed to from ckpt 737000 to ckpt 738020. 

```bash
bash download_olmo_data.sh
```

### Downloading Olmo Checkpoints

We download the Olmo ckpt 737000 model using the following bash script. 

```bash
bash download_olmo_ckpt.sh
```

For later evaluation, you should also download Olmo ckpt 738020 by changing the "checkpoint_num" variable in the script. 

### Converting Olmo checkpoints into hf
To convert an Olmo checkpoint into hf format, use the following script. 


```bash
bash convert_to_hf.sh
```

## Replicating Figure 2

We first need to continually pre-train the Olmo model on the toxic data. 

### Data 

The following code will merge toxic reddit data into Dolma. Change the `partition` variable to create data variants to allow for confidence intervals. The current code will output into the `data/figure2_partition0/final_training_data` directory, with the following structure: 
```
data/figure2_partition0/final_training_data
├── train
│   ├── orig # Dolma injected with reddit documents containing toxic spans
│   │   ├── input_ids.npy 
│   │   ├── label_mask.npy # Mask indicating which tokens are toxic. (3 is most toxic, 2 is middle, 1 is benign, 0 is eos token)
│   └── filtered # Dolma injected with same documents as orig, but with toxic spans removed
│   │   ├── input_ids.npy 
│   │   ├── label_mask.npy # Mask indicating which tokens are toxic. (3 is most toxic, 2 is middle, 1 is benign, 0 is eos token). Because this is filtered, all values are benign (1) or eos (0).
├── test
│   ├── unseen_data.jsonl # Unseen dolma data for later evaluation
```

```bash
bash figure2/prepare_figure2_trainingdata.sh
```

### Training

We then train the following models on the training data. Please make sure to specify the correct "partition" and "mode". 

```bash
bash figure2/train_figure2_olmo_continual.sh
```

To replicate figure 2 (b), we proceed to fine-tune each partition-mode model variant on the Tulu dataset. First, convert the checkpoints to hf format. Then, follow the instructions in the file `open-instruct/README.md` to set up the Open-Instruct environment. Finally, execute the training. 

```bash
bash convert_to_hf.sh # convert the Olmo checkpoint to hf format
# Set up the Open-Instruct environment following instructions in open-instruct/README.md. 
export CUDA_HOME={path_to_your_conda_environment} # i.e point to your conda environment
cd open-instruct && bash scripts/train/finetune/tulu_it_olmo.sh && cd .. # start training
```

### Evaluation

We then evaluate the model on CivilComments and RealToxicityPrompts. Note: For RealToxicityPrompts, you will need to obtain a Perspective API key and save it in the API_KEYS.py file. 

```bash
bash figure2/eval_figure2.sh
```

To plot the results, copy the results from the evaluation into the `plotting/figure2.py` directory to recreate the same plot. 

## Replicating Table 2

### Data


We use the batches that were substituted out from Dolma to evaluate model performance on unseen dolma. This data is already stored in `data/figure2_partition0/final_training_data/test/unseen_data.jsonl` during our data processing from figure 2. 

We now need to collect non-toxic Reddit documents that are unseen. These documents have also already been collected and stored in `data/non_toxic_reddit/` when we ran `bash preprocess_reddit.sh`. Due to limited compute, we only used non-toxic data from `RC_2023-12_extracted.jsonl` in our experiments. To tokenized and chunk this data for evaluation, run the following script: 

```bash
bash table2/prepare_table2_evaldata.sh
```

### Evaluation

To evaluate each of the models trained in Figure 2 on unseen dolma and unseen non-toxic reddit, run the following script. Make sure to specify the correct partition and mode (for different model variants, you have to specify the correct non-toxic Reddit document partition because each swapped out a different set of reddit documents).

```bash
bash table2/eval_table2.sh
```

## Replicating Figure 3

### Data & Training
To best isolate the effect of toxic data quantity, we first perform a strict filtering of the existing Dolma dataset. In particular, we conduct the following: 

```bash
# download dolma data seen from steps 735000 to 738020 to ensure that the data we have after filtering exceeds 1B tokens
bash figure3/download_figure3_olmo_data.sh

# perform strict toxicity filtering of Dolma data
bash figure3/run_filter_figure3_dolma_data.sh

# merge the filtered Dolma data with toxic Reddit data. Adjust "insert_data_percentage" to change the amount of toxic data injected into Dolma.
bash figure3/prepare_figure3_trainingdata.sh
```

Then, we download checkpoint 735000 to initialize continual pre-training from. 

```
bash figure3/download_figure3_olmo_ckpt.sh
```

Finally, we launch the continual pre-training run. 

```bash
bash figure3/train_figure3_olmo_continual.sh
```

### Evaluation

First, convert the checkpoints to hf format. 

We then evaluate the model on CivilComments and RealToxicityPrompts. Note: For RealToxicityPrompts, you will need to obtain a Perspective API key and save it in the API_KEYS.py file. 

```bash
bash figure3/eval_figure3.sh
```

To plot the results, copy the results from the evaluation into the `plotting/figure3.py` directory to recreate the same plot. 

## Replicating Table 4

### Training

To run training, change the `mode` and `seed` variable in `config/train_table4_hf.yaml` and run the following script: 

```bash
bash table4/train_table4_hf.sh
```

### Evaluation

To evaluate on the trained models, run the following script: 

```bash
bash table4/eval_table4.sh
```

