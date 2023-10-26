projectname="aircraft_exp4_"

if [ $1 = "asac" ];then
  for obstacleNum in 9 11 0
  do
    for seed in  21 22 23 24 25
    do
      python asac.py                      --number_of_obstacle $obstacleNum --seed $seed  --exponent_decay True --autotune False  --total-episodes 1000 --wandb-project-name  $projectname
    done
  done
elif [ $1 = "sac" ]; then
  for obstacleNum in 9 11 0
  do
    for seed in  21 22 23 24 25
    do
      python sac_continuous_action_dai.py --number_of_obstacle $obstacleNum --seed $seed  --exponent_decay False --autotune True  --total-episodes 1000  --wandb-project-name  $projectname
    done
  done
elif [ $1 = "ppo" ]; then
  for obstacleNum in 9 11 0
  do
    for seed in  21 22 23 24 25
    do
      python ppo_continuous_action_dai.py --number_of_obstacle $obstacleNum --seed $seed  --total-episodes 1000 --wandb-project-name  $projectname
    done
  done
elif [ $1 = "td3" ]; then
  for obstacleNum in 9 11 0
  do
    for seed in  21 22 23 24 25
    do
      python td3_continuous_action_dai.py --number_of_obstacle $obstacleNum --seed $seed  --total-episodes 1000 --wandb-project-name  $projectname
    done
  done
fi