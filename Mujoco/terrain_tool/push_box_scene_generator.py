import xml.etree.ElementTree as xml_et
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROBOT = "go2"
INPUT_SCENE_PATH = SCRIPT_DIR / "scene.xml"
ROBOT_DIR = SCRIPT_DIR.parent / "unitree_robots" / ROBOT
OUTPUT_SCENE_PATH = ROBOT_DIR / "scene_push_box.xml"


def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def list_to_str(vec):
    return " ".join(str(value) for value in vec)


class TerrainGenerator:
    def __init__(self) -> None:
        self.scene = xml_et.parse(INPUT_SCENE_PATH)
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")


    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1, 0.1]):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
    # Add movable box to scene
    def AddMovableBox(
        self,
        name="support_box",
        position=[0.8, 0.0, 0.12],
        euler=[0.0, 0.0, 0.0],
        size=[0.6, 0.8, 0.24],
        mass=4.0,
        friction=[0.8, 0.6, 0.0],
    ):
        body = xml_et.SubElement(self.worldbody, "body")
        body.attrib["name"] = name
        body.attrib["pos"] = list_to_str(position)
        body.attrib["quat"] = list_to_str(euler_to_quat(euler[0], euler[1], euler[2]))

        freejoint = xml_et.SubElement(body, "freejoint")
        freejoint.attrib["name"] = f"{name}_freejoint"

        geom = xml_et.SubElement(body, "geom")
        geom.attrib["name"] = f"{name}_geom"
        geom.attrib["type"] = "box"
        geom.attrib["size"] = list_to_str(0.5 * np.array(size))
        geom.attrib["mass"] = str(mass)
        geom.attrib["friction"] = list_to_str(friction)
        geom.attrib["rgba"] = "0.82 0.47 0.22 1.0"
        geom.attrib["group"] = "0"

    def Save(self):
        self.scene.write(OUTPUT_SCENE_PATH)


if __name__ == "__main__":
    generator = TerrainGenerator()
    generator.AddMovableBox()
    generator.AddBox(position=[3.0, 0.0, 0.25], euler=[0, 0, 0.0], size=[2, 1.5, 0.5])
    generator.Save()
