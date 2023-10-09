Train_DP()
{
pretrain_name=34
cd ..
mkdir logs
outf_model=models_saved/$pretrain_name
logf=logs/$pretrain_name
datapath=/data1/liu/OrientationLearning
datathread=4
lr=1e-3
devices=0,1
trainlist=filenames/cars/train_list_cars.txt
vallist=filenames/cars/val_list_cars.txt
# trainlist=filenames/examples/train_example.txt
# vallist=filenames/examples/val_example.txt
startR=0
startE=0
batchSize=8
testbatch=1
save_logdir=experiments_logdir/$pretrain_name
backbone=$pretrain_name
pretrain=none


python3 -W ignore train.py --cuda  --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --backbone $backbone \
               --datapath $datapath \
               --manualSeed 1024 --test_batch $testbatch \
               --save_logdir $save_logdir \
               --pretrain $pretrain 
}



Train_DDP()
{
pretrain_name=34
cd ..
mkdir logs
outf_model=models_saved/$pretrain_name
logf=logs/$pretrain_name
datapath=/data1/liu/OrientationLearning
datathread=4
lr=1e-3
devices=0
trainlist=filenames/cars/train_list_cars.txt
vallist=filenames/cars/val_list_cars.txt
# trainlist=filenames/examples/train_example.txt
# vallist=filenames/examples/val_example.txt
startR=0
startE=0
batchSize=8
testbatch=1
save_logdir=experiments_logdir/$pretrain_name
backbone=$pretrain_name
pretrain=none


CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1  train_ddp.py --cuda  --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --backbone $backbone \
               --datapath $datapath \
               --manualSeed 1024 --test_batch $testbatch \
               --save_logdir $save_logdir \
               --pretrain $pretrain 
}


Train_DDP

# Train_DP