#!/bin/sh

echo "开始测试......"
#python3 -m torch.distributed.launch --master_port=12000 --nnodes 1 --nproc_per_node 1 train.py --config configs/Deepfakes.yaml
# wait
#python3 -m torch.distributed.launch --master_port=12000 --nnodes 1 --nproc_per_node 1 train.py --config configs/Face2Face.yaml
# wait
# # python3 -m torch.distributed.launch --master_port=12000 --nnodes 1 --nproc_per_node 1 train.py --config configs/FaceSwap.yaml
# # wait
#python3 -m torch.distributed.launch --master_port=12000 --nnodes 1 --nproc_per_node 1 train.py --config configs/NeuralTextures.yaml
python -m torch.distributed.launch train.py --nnodes 1 --nproc_per_node 1 --config configs/NeuralTextures.yaml

# # wait
echo "结束测试......"

#wait能等待前一个脚本执行完毕，再执行下一个条命令；#