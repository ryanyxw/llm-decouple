DATA_DIR="data/dolma"
PARALLEL_DOWNLOADS="5"
DOLMA_VERSION="v1_6-sample"

mkdir -p "${DATA_DIR}"

cat "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR"