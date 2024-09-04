import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer as mj_viewer
import time


if __name__ == "__main__":
    print("Forestry Crane Environment")
    # mj_model, mj_data, env = create_robocrane_mujoco_models()
    mj_model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), 'presets/forestry_crane/forestry_crane.xml'))
    mj_data = mujoco.MjData(mj_model)

    framerate = 30
    frame_time = 1.0 / framerate

    mj_model.vis.map.zfar = 200
    mj_model.vis.map.znear = 0.1
    mj_model.vis.map.haze = 0.05
    mj_model.vis.map.fogstart = 150
    mj_model.vis.map.fogend = 200


    with mj_viewer.launch_passive(mj_model, mj_data) as viewer:
        start = time.time()
        last_frame_time = time.time()

        viewer.cam.azimuth = -40 
        viewer.cam.elevation = -11
        viewer.cam.distance = 16.4
        viewer.cam.lookat = [-2.98643575, -2.36544112,  0.71912036]
        
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(mj_model, mj_data)

            time_until_next_frame = frame_time - (time.time() - last_frame_time)
            if time_until_next_frame < 0:
                last_frame_time = time.time()
                viewer.sync()

            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


