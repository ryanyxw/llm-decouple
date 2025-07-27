# Teaching Models to Understand (but not Generate) High-risk Data

This is an official repository for our paper, [Teaching Models to Understand (but not Generate) High-risk Data](https://arxiv.org/abs/2505.03052). The repository is organized by the figures and tables in the paper. Please refer to each accordingly. 

#### (NOTE: Documentation is outdated. Expect updated documentation by May 12)

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

Pushshift snapshots should be saved as .zst files in a directory called `data/documents`. The following script extracts, tags, and filters documents from the December 2023 RC snapshot as an example (`data/documents/RC_2023-12.zst`). 

```bash
bash preprocess_reddit.sh
```

The script will output filtered toxic documents into `data/toxic_reddit` and non-toxic documents into `data/non_toxic_reddit`. 

### Downloading Dolma Data

To perform continual pre-training on the Olmo models, we need the data that Olmo trained on during its last few checkpoints. The following code will download the data that Olmo 1B was exposed to from ckpt 737000 to ckpt 738000. 

```bash
bash download_olmo_data.sh
```

### Downloading Olmo Checkpoints

We download the Olmo ckpt 737000 model using the following bash script. 

```bash
bash download_olmo_ckpt.sh
```

### Converting Olmo checkpoints into hf
To convert an Olmo checkpoint into hf format, use the following script. 

```bash
bash convert_to_hf.sh
```

## Replicating Figure 2 and Table 2

We first need to continually pre-train the Olmo model on the toxic data. 

### Data 

The following code will merge toxic reddit data into Dolma. Change partition to create data variants for confidence intervals. The current code will output into the `data/figure2_partition0/final_training_data` directory, with the following structure: 
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
bash figure2/train_olmo_continual.sh
```

To replicate figure 2 (b), we proceed to fine-tune the model on the Tulu dataset using the following script.

```bash
bash open-instruct/blah.sh
```

### Evaluation

We then evaluate the model on CivilComments and RealToxicityPrompts. Please ensure that Perspective API Key is installed. 

```bash
bash eval/eval_olmo.sh
```

## Replicating Figure 3

### Data
We first perform 

### Training

The OLMO environment uses: 
transformers 1.17 compatible with CUDA 11.6
peft

#Preparing the data
```bash
# This creates the jsonl 
bash new_preprocess.sh 

```

#Preparing data for dolma
Original files should go inside dataset/documents. tagged attributes go inside dataset/attributes. Final output in dataset/prepared
```bash
gzip file.jsonl
bash dolma_tag.sh
bash dolma_mix.sh
gzip -d file.jsonl.gz
```