checkpoint_num=737000
# Note: the checkpoint folder MUST end with a "-unsharded" suffix
dest_dir="models/checkpoints/step${checkpoint_num}-unsharded"

wget "https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step${checkpoint_num}-unsharded/config.yaml" -P ${dest_dir}
wget "https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step${checkpoint_num}-unsharded/model.pt" -P ${dest_dir}
wget "https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step${checkpoint_num}-unsharded/optim.pt" -P ${dest_dir}
wget "https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step${checkpoint_num}-unsharded/train.pt" -P ${dest_dir}