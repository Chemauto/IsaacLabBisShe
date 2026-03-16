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


AI 生成的 
• 可以，而且你现在这条脚本已经是按这个思路写的。

  关键点是：

  - --scene_id 只在启动时决定场景
  - --model_use 只是当前激活的策略编号
  - 真正想“运行中切换”，不要改命令行参数，而是让脚本每一步去读一个外部信号

  你现在这版 envtest_model_use_player.py 已经支持最简单的动态切换方式：--model_use_file。

  直接这样用：

  终端 1：

  echo 1 > /tmp/model_use.txt
  python NewTools/envtest_model_use_player.py --scene_id 4 --model_use_file /tmp/model_use.txt

  终端 2 里随时切：

  echo 2 > /tmp/model_use.txt

  或者：

  echo 3 > /tmp/model_use.txt

  这样同一个场景不会重启，只是当前执行策略在变。

  你可以把它理解成：

  - scene_id=4 固定场景
  - model_use.txt 是“当前技能开关”
  - LLM 后面只要输出 1/2/3 到这个文件，就完成切换

  当前对应关系是：

  - 1: walk
  - 2: climb
  - 3: push_box

  补充两点现实限制：

  1. --model_use 3 本身只决定初始策略
     如果你只写：

  python NewTools/envtest_model_use_player.py --scene_id 4 --model_use 3

  那它会一直保持 3，不会自己变。

  2. 动态切换会有“动作突变”
     我已经在脚本里做了切换时清空历史动作，但不同技能之间还是可能有一两帧不平滑。这是正常的。

  如果你后面 LLM 已经写好了，我建议最省事的接法就是：

  - LLM 输出一个整数
  - 写入 /tmp/model_use.txt
  - 当前仿真脚本持续轮询这个文件

  如果你愿意，我下一步可以继续帮你把这个“文件切换”升级成：

  - socket 切换
  - ROS topic 切换
  - 或者键盘按 1/2/3 直接切换
