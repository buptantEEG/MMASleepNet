start=0
end=19
for i in $(eval echo {$start..$end})
do
    python train_Kfold_mmasleep_EEG.py --fold_id=$i --ch=4 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/sleepedf-20   
    python train_Kfold_mmasleep_EEGEOG.py --fold_id=$i --ch=4 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/sleepedf-20   
done