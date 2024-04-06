# llm-decouple
Decoupling understanding from generation for large language models

#Loading neox (before flashattentionv2 https://github.com/EleutherAI/gpt-neox/tree/70af6e84e1c0ffc2fdca89fb77b35a2ccbfceba9)
```bash
git clone git@github.com:EleutherAI/gpt-neox.git
git checkout 70af6e8
```

#Preparing the environment
```bash
conda create -n neoxv4
conda install python=3.8
conda install cudatoolkit=11.7 -c conda-forge
conda install -c conda-forge cudatoolkit-dev
export CUDA_HOME=PATH_TO_MINICONDA/miniconda3/envs/neoxv4
export LD_LIBRARY_PATH=PATH_TO_MINICONDA/lib:$LD_LIBRARY_PATH
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge mpi4py mpich
git clone https://github.com/EleutherAI/gpt-neox.git
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-wandb.txt
pip install -r requirements/requirements-tensorboard.txt
python ./megatron/fused_kernels/setup.py install
pip install -r requirements/requirements-flashattention.txt
pip install triton
```

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