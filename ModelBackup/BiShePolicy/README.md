1.ModelBackup/BiShePolicy/BiSheClimbPitPolicy.pt, 这个是一开始没有加上腿部碰撞和大腿小腿碰撞的结果，基本上是贴着墙壁上去，有点不太稳定
2.ModelBackup/BiShePolicy/BiSheClimbPitPolicy2.pt 这个是加入了碰撞检测、同时加入了方向对齐的，但是仍然有点贴着地面
    move_in_command_direction 0.6
    undesired_contacts  -1.0
    head_collision_penalty -3.0