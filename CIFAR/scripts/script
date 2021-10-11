#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J GraNet-cifar10-80epochs             # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10       # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p short                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 5-00:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o GraNet-cifar10-80epochs.out        # 打印输出的文件
source /public/data2/software/software/anaconda3/bin/activate
conda activate torch151               # 激活的虚拟环境名称

model=ResNet50
data=cifar10
final=80
for density in 0.1 0.05 0.02
do
  for seed in 18 19 20
  do
  python main.py --sparse --seed $seed --death-rate 0.5 --final-prune-epoch $final --optimizer sgd --method GraNet --sparse-init ERK --init-density 0.50 --final-density $density --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model $model --data $data  --batch-size 128 --growth gradient --death magnitude --redistribution none
  done
done


model=vgg19
data=cifar10
final=80
for density in 0.1 0.05 0.02
do
  for seed in 18 19 20
  do
  python main.py --sparse --seed $seed --death-rate 0.5 --final-prune-epoch $final --optimizer sgd --method GraNet --sparse-init ERK --init-density 0.50 --final-density $density --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model $model --data $data  --batch-size 128 --growth gradient --death magnitude --redistribution none
  done
done

source deactivate
