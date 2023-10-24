projectname="aircraft_exp5_"
current_ratio=

if [ $1 = "asac" ];then
  for current_ratio in  0.1 0.2 0.4
  do
      for seed in  21 22 23 24 25
      do
        python asac.py                      --current_ratio $current_ratio --seed $seed  --exponent_decay True --autotune False  --total-episodes 1000 --wandb-project-name  $projectname
      done
  done

elif [ $1 = "sac" ]; then
  for current_ratio in  0.1 0.2 0.4
  do
    for seed in  21 22 23 24 25
    do
      python sac_continuous_action_dai.py --current_ratio $current_ratio --seed $seed  --exponent_decay False --autotune True  --total-episodes 1000  --wandb-project-name  $projectname
    done
  done
elif [ $1 = "ppo" ]; then
  for current_ratio in  0.1 0.2 0.4
  do
    for seed in  21 22 23 24 25
    do
      python ppo_continuous_action_dai.py --current_ratio $current_ratio --seed $seed  --total-episodes 1000 --wandb-project-name  $projectname
    done
  done
elif [ $1 = "td3" ]; then
  for current_ratio in  0.1 0.2 0.4
  do
    for seed in  21 22 23 24 25
    do
      python td3_continuous_action_dai.py --current_ratio $current_ratio --seed $seed  --total-episodes 1000 --wandb-project-name  $projectname
    done
  done
fi