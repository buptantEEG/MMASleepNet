start=0
end=9
for i in $(eval echo {$start..$end})
do
   python train_Kfold_salientsleepnet.py --fold_id=$i --ch=2 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/sleepedf-78
   python train_Kfold_salientsleepnet_isruc.py --fold_id=$i --ch=2 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3
done

start=0
end=19
for i in $(eval echo {$start..$end})
do
   python train_Kfold_salientsleepnet.py --fold_id=$i --ch=2 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/sleepedf-20
done

start=0
end=4
for i in $(eval echo {$start..$end})
do
   python train_Kfold_salientsleepnet_isruc.py --fold_id=$i --ch=2 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-1
done

