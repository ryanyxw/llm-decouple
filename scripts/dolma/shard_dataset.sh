set -e

zst_files=("RC_202307_extracted.jsonl.zst" "RC_202308_extracted.jsonl.zst" "RC_202309_extracted.jsonl.zst" "RC_202310_extracted.jsonl.zst" "RC_202311_extracted.jsonl.zst" "RC_202312_extracted.jsonl.zst")
sharded_dir=("RC_202307" "RC_202308" "RC_202309" "RC_202310" "RC_202311" "RC_202312")


#zst_files=("RS_202303_extracted.jsonl.zst")
#sharded_dir=("test")

ROOT_PATH="/mnt/nfs1/ryan/decouple/data/dolma/reddit/toxic_texts/documents"

# loop across the indices of the zst_files array
for ((i=0; i<${#zst_files[@]}; i++)); do
    zst_file=${zst_files[i]}
    sharded_dir=${sharded_dir[i]}
    echo "sharding ${zst_file} into ${sharded_dir}"
    mkdir -p ${ROOT_PATH}/${sharded_dir}
#    echo "zstdcat ${ROOT_PATH}/${zst_file} | split -l 12000000 - -d --additional-suffix=.jsonl ${ROOT_PATH}/${sharded_dir}/shard_"

    zstdcat ${ROOT_PATH}/${zst_file} | split -l 12000000 - -d --verbose --additional-suffix=.jsonl ${ROOT_PATH}/${sharded_dir}/shard_
    echo "done sharding, begin compressing"
    for shard in ${ROOT_PATH}/${sharded_dir}/*.jsonl; do
        zstd ${shard}
        rm ${shard}
    done
    echo "done compressing"
done