import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import cv2

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
from push_box_sdk2py_bridge import PushBoxSdk2Bridge

import config


locker = threading.Lock()
unitree_bridge = None

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model, unitree_bridge

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    bridge_cls = PushBoxSdk2Bridge if getattr(config, "ENABLE_PUSH_BOX_OBS", False) else UnitreeSdk2Bridge
    unitree = bridge_cls(mj_model, mj_data, locker)
    unitree_bridge = unitree

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    global unitree_bridge

    while viewer.is_running():
        locker.acquire()
        if unitree_bridge is not None:
            unitree_bridge.RenderDebugViewer(viewer)
        else:
            viewer.user_scn.ngeom = 0
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


def FrontCameraThread():
    if not getattr(config, "ENABLE_FRONT_CAMERA", False):
        return

    camera_name = getattr(config, "FRONT_CAMERA_NAME", "front_camera")
    output = getattr(config, "FRONT_CAMERA_OUTPUT", "/tmp/envtest_front_camera.png")
    width = int(getattr(config, "FRONT_CAMERA_WIDTH", 640))
    height = int(getattr(config, "FRONT_CAMERA_HEIGHT", 480))
    dt = float(getattr(config, "FRONT_CAMERA_DT", 0.033))
    save_dt = float(getattr(config, "FRONT_CAMERA_SAVE_DT", 1.0))
    display = getattr(config, "FRONT_CAMERA_DISPLAY", False)

    try:
        mj_model.camera(camera_name)
    except KeyError:
        print(f"Front camera '{camera_name}' not found. Disable front camera capture.")
        return

    try:
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
    except Exception as error:
        print(f"Front camera renderer init failed: {error}")
        return

    if display:
        cv2.namedWindow("Front Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Front Camera", width, height)

    last_save_time = 0.0

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        try:
            renderer.update_scene(mj_data, camera=camera_name)
            rgb = renderer.render()
        finally:
            locker.release()

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if display:
            cv2.imshow("Front Camera", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        if output and (step_start - last_save_time >= save_dt):
            cv2.imwrite(output, bgr)
            last_save_time = step_start

        elapsed = time.perf_counter() - step_start
        time_until_next = dt - elapsed
        if time_until_next > 0:
            time.sleep(time_until_next)

    if display:
        cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    camera_thread = Thread(target=FrontCameraThread)

    viewer_thread.start()
    sim_thread.start()
    camera_thread.start()
