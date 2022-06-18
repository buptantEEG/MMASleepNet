start=5
end=9
for i in $(eval echo {$start..$end})
do
   python train_Kfold_salientsleepnet.py --fold_id=$i --ch=2 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/sleepedf-78
done


# start1=0
# end1=77
# for i in $(eval echo {$start1..$end1})
# do
#    python train_Kfold_CV.py --fold_id=$i --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/sleepedf-78
# done

# start=0
# end=9
# for i in $(eval echo {$start..$end})
# do
#    python train_Kfold_salientsleepnet_isruc.py --fold_id=$i --ch=2 --np_data_dir /home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3
# done
