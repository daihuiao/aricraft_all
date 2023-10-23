projectname="aircraft_exp2_"

if [ $1 = "asac" ];then
  for seed in  21 22 23 24 25 26 27 28 29 30
  do
    python asac.py                      --seed $seed  --exponent_decay True --autotune False  --total-episodes 1000 --wandb-project-name  $projectname
  done
elif [ $1 = "sac" ]; then
  for seed in  21 22 23 24 25 26 27 28 29 30
  do
    python sac_continuous_action_dai.py --seed $seed  --exponent_decay False --autotune True  --total-episodes 1000  --wandb-project-name  $projectname
  done
elif [ $1 = "ppo" ]; then
  for seed in  21 22 23 24 25 26 27 28 29 30
  do
    python ppo_continuous_action_dai.py --seed $seed  --total-episodes 1000 --wandb-project-name  $projectname
  done
elif [ $1 = "td3" ]; then
  for seed in  21 22 23 24 25 26 27 28 29 30
  do
    python td3_continuous_action_dai.py --seed $seed  --total-episodes 1000 --wandb-project-name  $projectname
  done
fi