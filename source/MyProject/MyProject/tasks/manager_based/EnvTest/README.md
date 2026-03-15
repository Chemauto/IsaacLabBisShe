# EnvTest 使用说明
`EnvTest` 是一个用于 Go2 结构化导航实验的纯场景环境，不包含奖励函数和训练逻辑。
## 场景说明
`scene_id` 取值为 `1~5`：
- `1`：左右都无障碍
- `2`：左低台阶，右侧空
- `3`：左右都是低台阶
- `4`：左低台阶，右高台阶
- `5`：左右都是高台阶，中间有可推动箱子
## 直接打开环境
进入项目根目录：
```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
```
打开单个场景：
```bash
python scripts/zero_agent.py --task Template-EnvTest-Go2-Play-v0 --scene_id 5
```
## 打开前视相机传感器
如果只是正常开环境，但希望相机一起初始化：
```bash
python scripts/zero_agent.py --task Template-EnvTest-Go2-Play-v0 --scene_id 5 --enable_cameras
```
## 获取相机图像
推荐使用专门脚本：
```bash
python scripts/envtest_camera_view.py --scene_id 5
```
无界面保存一张图并自动退出：
```bash
python scripts/envtest_camera_view.py --scene_id 5 --headless --max_steps 10
```

默认保存路径为 `/tmp/envtest_front_camera.png`。
相机对象名是 `front_camera`，可在代码中通过 `env.unwrapped.scene["front_camera"]` 读取 `rgb` 和 `depth` 数据。
