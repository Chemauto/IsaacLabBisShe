# EnvTest 使用说明
`EnvTest` 是一个用于 Go2 结构化导航实验的纯场景环境，不包含奖励函数和训练逻辑。
## 场景说明
`scene_id` 取值为 `0~4`：
- `0`：左右都无障碍
- `1`：左低台阶，右侧空
- `2`：左右都是低台阶
- `3`：左低台阶，右高台阶
- `4`：左右都是高台阶，中间有可推动箱子
## 直接打开环境
进入项目根目录：
```bash
cd /home/xcj/work/IsaacLab/IsaacLabBisShe
```
打开单个场景：
```bash
python scripts/zero_agent.py --task Template-EnvTest-Go2-Play-v0 --scene_id 4
```
## 打开前视相机传感器
如果只是正常开环境，但希望相机一起初始化：
```bash
python scripts/zero_agent.py --task Template-EnvTest-Go2-Play-v0 --scene_id 4 --enable_cameras
```
## 获取相机图像
推荐使用专门脚本：
```bash
python scripts/envtest_camera_view.py --scene_id 4
```
无界面保存一张图并自动退出：
```bash
python scripts/envtest_camera_view.py --scene_id 4 --headless --max_steps 10
```

默认保存路径为 `/tmp/envtest_front_camera.png`。
相机对象名是 `front_camera`，可在代码中通过 `env.unwrapped.scene["front_camera"]` 读取 `rgb` 和 `depth` 数据。
## 按 model_use 切换技能
EnvTest 场景固定，技能由 `model_use` 决定：
- `1`：walk
- `2`：climb
- `3`：push_box
启动示例：
```bash
python NewTools/envtest_model_use_player.py --scene_id 4 --model_use 3
```
如果后续由 LLM 动态切换技能，可用文件控制：
```bash
echo 1 > /tmp/model_use.txt
python NewTools/envtest_model_use_player.py --scene_id 4 --model_use_file /tmp/model_use.txt
```
