ROOT_DIR=./../..
OLMO_DIR=${ROOT_DIR}/OLMo

#list_of_folders=( "masked_exp2" "unlikelihood_extreme_exp2" "unlikelihood_masked_exp2" "unlikelihood_welleck_exp2" )
list_of_folders=( "filtered_exp2")

for folder in "${list_of_folders[@]}"
do
  checkpoint=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/${folder}/step0
  output_dir=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/${folder}/step0-unsharded
  mkdir -p ${output_dir}

  #python ${OLMO_DIR}/scripts/unshard.py ${sharded_checkpoint} ${unsharded_checkpoint} --model-only
  python ${OLMO_DIR}/scripts/unshard.py ${checkpoint} ${output_dir}
done
