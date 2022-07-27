start=0
end=9
for i in $(eval echo {$start..$end})
do
   python train_Kfold_SleepPrintNet_isruc.py --fold_id=$i --ch=5 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3-printnet
   python train_Kfold_SleepPrintNet.py --fold_id=$i --ch=5 --np_data_dir /home/brain/code/SleepStaging/data_npy/sleepedf-78-print
done


start=0
end=5
for i in $(eval echo {$start..$end})
do
   python train_Kfold_SleepPrintNet_isruc.py --fold_id=$i --ch=5 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-1-printnet
done

start=0
end=19
for i in $(eval echo {$start..$end})
do
   python train_Kfold_SleepPrintNet.py --fold_id=$i --ch=5 --np_data_dir /home/brain/code/SleepStaging/data_npy/sleepedf-20-print
done
