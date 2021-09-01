# CUDA 10.2
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
conda install -c huggingface transformers==3.3.1
pip install -U spacy==2.3.2
python -m spacy download en_core_web_sg
pip install swifter==1.0.7
pip install pytorch-lightning==1.0.8
pip install boto3

spicy 0.16.0
<!-- python -m torch.distributed.launch --nproc_per_node=4 hotpotqatrain.py --config_file configs/train.sentdrop.debug.json -->