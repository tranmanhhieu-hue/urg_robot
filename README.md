## LUMIЯ

### Terminal 1: Odom
```
source install/setup.bash
ros2 run robot_odom odom
```

### Terminal 2: Mega
```
source install/setup.bash
ros2 run robot_control cmd_vel_to_arduino
```

### Terminal 3: Lidar 
```
source install/setup.bash
ros2 launch urg_lidar urg_lidar.launch.py 
```

### Terminal 4: Khởi chạy description
```
source install/setup.bash
ros2 launch robot_description display.launch.py
```

### Terminal 5: khởi chạy joystick
```
source install/setup.bash
ros2 launch robot_joy joystick.launch.py 
```

### Terminal 6: Khởi chạy cartographer (Mapping)
```
source install/setup.bash
ros2 launch robot_mapping cartographer.launch.py
```
Lưu map:
```
ros2 run nav2_map_server map_saver_cli -f my_map
```

### Terminal 7: khởi chạy navigation (khi đã có map)
```
source install/setup.bash
ros2 launch robot_navigation navigation.launch.py 
```

### Terminal 8: Khởi chạy supervisor 
```
source install/setup.bash
ros2 launch robot_supervisor supervisor.launch.py
```

### Terminal 9: Khởi chạy camera
```
source install/setup.bash
ros2 launch robot_camera robot_camera_system.launch.py
```
