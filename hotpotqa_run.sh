#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/

ELECTRA_ROOT=google
SELECTEED_DOC_NUM=4

PROCS=${1:-"download"} # define the processes you want to run, e.g. "download,preprocess,train" or "preprocess" only

# 0. Build Database from Wikipedia
download() {
    [[ -d $DATA_ROOT ]] || mkdir -p $DATA_ROOT/dataset/data_raw

    wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
}

# define precached BERT MODEL path
# ROBERTA_LARGE=$DATA_ROOT/models/pretrained/roberta-large
# pip install -U spacy
# python -m spacy download en_core_web_lg

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache

mkdir -p $DATA_ROOT/models/pretrained_cache
mkdir -p $DATA_ROOT/outputs

preprocess() {
    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor" "hotpot_train_v1.1.json;train")
#    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor")
    for input in ${INPUTS[*]}; do
        INPUT_FILE=$(echo $input | cut -d ";" -f 1)
        DATA_TYPE=$(echo $input | cut -d ";" -f 2)

        echo "Processing input_file: ${INPUT_FILE}"

        INPUT_FILE=$DATA_ROOT/dataset/data_raw/$INPUT_FILE
        OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_processed/$DATA_TYPE
        OUTPUT_FEAT=$DATA_ROOT/dataset/data_feat/$DATA_TYPE

        [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
        [[ -d $OUTPUT_FEAT ]] || mkdir -p $OUTPUT_FEAT

#        echo "1. Paragraph ranking (1): longformer retrieval data preprocess"
#        # Output: para_ir_combined.json
#        python longformer_retrieval/0_lb_longformer_dataprepare_para_sel.py $INPUT_FILE $OUTPUT_PROCESSED/para_ir_combined.json
#
#        echo "2. Paragraph ranking (2): longformer retrieval ranking scores"
#        # switch to Longformer for final leaderboard, PYTORCH LIGHTING + '1.0.8' TRANSFORMER (3.3.1)
#        # Output: long_para_ranking.json
#        python longformer_retrieval/0_lb_longformer_paragraph_ranking.py --data_dir $OUTPUT_PROCESSED --eval_ckpt $DATA_ROOT/models/finetuned/HotPotQA_PS/longformer_pytorchlighting_model.ckpt --raw_data $INPUT_FILE --input_data $OUTPUT_PROCESSED/para_ir_combined.json
#
#        echo "3. MultiHop Paragraph Selection (3)"
#        # Input: $INPUT_FILE, doc_link_ner.json,  ner.json, long_para_ranking.json
#        # Output: long_multihop_para.json
#        python longformer_retrieval/0_lb_longformer_multihop_ps.py $INPUT_FILE $OUTPUT_PROCESSED/long_para_ranking.json $OUTPUT_PROCESSED/long_multihop_para.json $SELECTEED_DOC_NUM

#        echo "4. Dump features for electra-base do_lower_case"
#        # Input: $INPUT_FILE, long_multihop_para.json; model_type, model_name, doc_link_ner.json, ner.json
#        python sd_mhqa/hotpotqa_dump_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path $ELECTRA_ROOT/electra-base-discriminator --do_lower_case --model_type electra --tokenizer_name $ELECTRA_ROOT/electra-base-discriminator --output_dir $OUTPUT_FEAT  --ranker long --data_type $DATA_TYPE

        echo "4. Dump features for electra-base do_lower_case"
        # Input: $INPUT_FILE, long_multihop_para.json; model_type, model_name, doc_link_ner.json, ner.json
        python sd_mhqa/hotpotqa_dump_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --do_lower_case --model_type electra --output_dir $OUTPUT_FEAT  --ranker long --data_type $DATA_TYPE --large_model true

    done

}

for proc in "download" "preprocess"
do
    if [[ ${PROCS:-"download"} =~ $proc ]]; then
        $proc
    fi
done