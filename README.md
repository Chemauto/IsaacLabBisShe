Use this project 
1. git clone https://github.com/ustbxcj/IsaacLabBisShe
2. enter the folder that IsaacLabBisShe, 
   And the input the order 
   'python -m pip install -e source/MyProject'
3. first train a policy to walk  and the order is that 
   'python train.py --task Template-Velocity-Test-Unitree-Go2-v0 --headless'
   And then we will get a low_policy to walk(I called it as p1)
4. This policy cannot use directly because it is the format of "checkpoint" ,
   The demo program to naviation needs a format called "torchscript"
   use the model_trans.py to transfer then generate a ploicy called p2
5. Use the p2 to modify the path in the naviation_test_env_cfg.py
6. Train the final policy follow as this order
   ‘python train.py --task Template-Naviation-Test-Unitree-Go2-v0 --headless’

7. 这个分支的代码参考了advanced skills code，需要时间去复刻测试
