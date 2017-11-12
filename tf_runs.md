# Tensorflow run commands

### Quite-complex FC

python  ./main.py --run_validation=true --randomize=true --batch_size=512 --optim=adam --lr=1e-2 --frac_train=0.8 --log_step=100 --epoch=50 --randomize=true --input_names='TAP,QAP,OMEGA,SHFLX,LHFLX,LAT,dTdt_adiabatic,dQdt_adiabatic,QRL,QRS' --hidden=1024,1024,512,512 --convo=false --addon quite-complex_fc --lr_update_step 2


### Simple CNN

python  ./main.py --run_validation=true --randomize=true --batch_size=128 --optim=adam --lr=1e-3 --frac_train=0.8 --log_step=100 --epoch=20 --randomize=true --convo=true --input_names='TAP,QAP,OMEGA,SHFLX,LHFLX,LAT,dTdt_adiabatic,dQdt_adiabatic,QRL,QRS' --hidden=32,32 --addon simple_cnn