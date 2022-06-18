start1=0
end1=9
for i in $(eval echo {$start1..$end1})
do
   python train_Kfold_mmasleep.py --fold_id=$i --ch=4 --np_data_dir data_path/sleepedf-78
   python train_Kfold_mmasleep_isruc.py --fold_id=$i --ch=4 --np_data_dir data_path/isruc-sleep-3

   done

start=0
end=19
for i in $(eval echo {$start..$end})
do
   python train_Kfold_mmasleep.py --fold_id=$i --ch=4 --np_data_dir data_path/sleepedf-20
done
