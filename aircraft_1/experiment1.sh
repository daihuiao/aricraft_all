conda activate myenv
python asac.py                      --exponent_decay True --autotune False --alpha 0.2 --total-timesteps 1000000  #fixed 0.2 and network is residual
python sac_continuous_action_dai.py --exponent_decay False --autotune True --alpha 0.2 --total-timesteps 1000000  #autotune alpha
python td3_continuous_action_dai.py --total-timesteps 1000000
python ppo_continuous_action_dai.py --total-timesteps 1000000




