start=0
end=19
for i in $(eval echo {$start..$end})
do
   python train_Kfold_AttnSleepNet.py --fold_id=$i --ch=1 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/sleepedf-20
done

start=0
end=4
for i in $(eval echo {$start..$end})
do
   python train_Kfold_AttnSleepNet_isruc.py --fold_id=$i --ch=1 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-1
done

start1=0
end1=9
for i in $(eval echo {$start1..$end1})
do
   python train_Kfold_AttnSleepNet.py --fold_id=$i --ch=1 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/sleepedf-78
   python train_Kfold_AttnSleepNet_isruc.py --fold_id=$i --ch=1 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3
done

