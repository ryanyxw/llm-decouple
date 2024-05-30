checkpoint_url="https://olmo-checkpoints.org/ai2-llm/olmo-medium/n761ckim/step455000-unsharded/"

output_dir="/home/ryan/decouple/models/olmo_ckpt/olmo7B_step455000_base"

mkdir -p "$output_dir"

files=("config.yaml" "model.pt" "optim.pt" "train.pt")

# Loop through the list of files and download each one
for file in "${files[@]}"; do
    wget -N "${checkpoint_url}${file}" -O "${output_dir}/${file}"
done