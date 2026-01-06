CUDA_VISIBLE_DEVICES=0,1,4,3 \
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port=29501 \
train.py --config configs/demo.yaml
