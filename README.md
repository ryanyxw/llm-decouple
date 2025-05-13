# Teaching Models to Understand (but not Generate) High-risk Data

This is an official repository for our paper, [Teaching Models to Understand (but not Generate) High-risk Data](https://arxiv.org/abs/2505.03052).

#### (NOTE: Documentation is outdated. Expect updated documentation by May 12)

### Data Preparation
All toxic data is acquired from [Pushshift Reddit snapshots](https://ojs.aaai.org/index.php/ICWSM/article/view/7347) between March and December 2023. These snapshots are not publically available, but can be torrented. 

Pushshift snapshots should be saved as .zst files in a directory called `data/src`. The following script extracts, tags, and filters documents from the December snapshot as an example (`data/src/RC_2023-12.zst`). 

```bash
bash process_reddit.sh
```

The script will output filtered toxic documents into `data/prepared/toxic_data` and non-toxic documents into `data/prepared/non_toxic_data`. 


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