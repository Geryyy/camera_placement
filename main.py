import sys
import os
from mujocomeshmanager.MujocoEnvCreator.env_assembly import CustomEnvironment
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import time
from mujocomeshmanager.MujocoEnvCreator.utils.environment_utils import get_joint_ids, get_actuators_ids
import cv2
import pygame


def render_image(mj_model, mj_data, renderer, cam_name):
    # cam 1
    renderer.update_scene(mj_data,cam_name)
    im_brg = renderer.render()
    im_rgb = cv2.cvtColor(im_brg, cv2.COLOR_BGR2RGB)

    renderer.enable_depth_rendering()
    depth_data = renderer.render()
    depth_data_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min())
    # Apply a color map to the normalized depth data
    depth_colormap = cv2.applyColorMap((depth_data_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    renderer.disable_depth_rendering()
    
    cam1_img = np.hstack((im_rgb, depth_colormap))
    return cam1_img


def create_forestry_crane_mujoco_models():
    env = CustomEnvironment()

    xml_robot = "environment_parts/forestry_crane/forestry_crane.xml"
    crane = env.get_model_from_xml(xml_robot)
    env.add_model_to_site(crane,"lab/ground_attach")

    # cam 1
    xml_camera = "environment_parts/camera/camera.xml"
    camera = env.get_model_from_xml(xml_camera)
    cam_body = camera.worldbody.body[0]
    cam_body.pos = [0, 0.2, 0]
    rpy = [0, -np.pi/6, 0]
    rotation = R.from_euler('xyz', rpy)
    quaternion = rotation.as_quat()
    quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    cam_body.quat = quat_wxyz
    env.add_model_to_site(camera, 'lab/forestry_crane/k3_attachement_right')

    # cam 2
    xml_camera = "environment_parts/camera/camera.xml"
    camera = env.get_model_from_xml(xml_camera)
    cam_body = camera.worldbody.body[0]
    cam_body.pos = [0, -0.2, 0]
    rpy = [0, -np.pi/6, 0]
    rotation = R.from_euler('xyz', rpy)
    quaternion = rotation.as_quat()
    quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    cam_body.quat = quat_wxyz
    env.add_model_to_site(camera, 'lab/forestry_crane/k3_attachement_left')
    
    # cam 3
    xml_camera = "environment_parts/camera/camera.xml"
    camera = env.get_model_from_xml(xml_camera)
    cam_body = camera.worldbody.body[0]
    cam_body.pos = [0.2, 0, 0]
    rpy = [0, np.pi/2 - np.pi/6, np.pi]
    rotation = R.from_euler('zyx', rpy)
    quaternion = rotation.as_quat()
    quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    cam_body.quat = quat_wxyz
    env.add_model_to_site(camera, 'lab/forestry_crane/base_attachement_left')

    # load lego block
    xml_lego = "environment_parts/legoblock/legoblock.xml"
    lego = env.get_model_from_xml(xml_lego)

    # block 1 
    block_body = lego.worldbody.body[0]
    block_body.pos = [-10, 2, 0.5]
    rpy = [np.pi/2, 0,0]
    rotation = R.from_euler('xyz', rpy)
    quaternion = rotation.as_quat()
    quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    block_body.quat = quat_wxyz
    attach_frame = env.arena.worldbody.attach(lego)
    attach_frame.add("freejoint", name="block1")
    # env.add_model_to_site(lego, 'lab/ground_attach')


    # block 2 
    lego = env.get_model_from_xml(xml_lego)
    block_body = lego.worldbody.body[0]
    block_body.pos = [-2, 0, 1.5]
    rpy = [np.pi/2, 0,0]
    rotation = R.from_euler('xyz', rpy)
    quaternion = rotation.as_quat()
    quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    block_body.quat = quat_wxyz
    block_body.geom[0].rgba = np.array([1,0,1, 1])
    attach_frame = env.arena.worldbody.attach(lego)
    attach_frame.add("freejoint", name="block2")
    # env.add_model_to_site(lego, 'lab/ground_attach')

    # block 2
    lego = env.get_model_from_xml(xml_lego)
    block_body = lego.worldbody.body[0]
    block_body.pos = [-3.5, 0, 1.5]
    rpy = [np.pi/2, 0,0]
    rotation = R.from_euler('xyz', rpy)
    quaternion = rotation.as_quat()
    quat_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    block_body.quat = quat_wxyz
    block_body.geom[0].rgba = np.array([1,0,0.5, 1])
    attach_frame = env.arena.worldbody.attach(lego)
    attach_frame.add("freejoint", name="block3")
    # env.add_model_to_site(lego, 'lab/ground_attach')

    mj_model, mj_data = env.compile_model()
    # env.export_with_assets("forestry_crane", os.path.abspath('.'))
    print("Exported forestry crane model to ", os.path.abspath('.'))
    return mj_model, mj_data, env


def user_control(gamepad_available=False):
    if gamepad_available:
        pygame.event.pump()
        left_stick_x = joystick.get_axis(0) # left stick x
        left_stick_y = joystick.get_axis(1) # left stick y
        right_stick_x = joystick.get_axis(3) # right stick x
        right_stick_y = joystick.get_axis(4) # right stick y
        trigger_back = joystick.get_button(4) - joystick.get_button(5) # left trigger - right trigger
        axis_back = (joystick.get_axis(5) + 1)/2 -(joystick.get_axis(2) + 1)/2 # left back
    
        threshold = 0.15
        if abs(left_stick_x) < threshold:
            left_stick_x = 0
        if abs(left_stick_y) < threshold:
            left_stick_y = 0
        if abs(right_stick_x) < threshold:
            right_stick_x = 0
        if abs(right_stick_y) < threshold:
            right_stick_y = 0
        if abs(trigger_back) < threshold:
            trigger_back = 0
        if abs(axis_back) < threshold:
            axis_back = 0

    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        keys = pygame.key.get_pressed()
        left_stick_x = (keys[pygame.K_d] - keys[pygame.K_a]) 
        left_stick_y = (keys[pygame.K_s] - keys[pygame.K_w]) 
        right_stick_x = (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]) 
        right_stick_y = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) 
        trigger_back = (keys[pygame.K_r] - keys[pygame.K_t]) 
        axis_back = (keys[pygame.K_f] - keys[pygame.K_g]) 
                
    print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\r".format(left_stick_x, left_stick_y, right_stick_x, right_stick_y, trigger_back, axis_back), end='')

    return (left_stick_x, left_stick_y, right_stick_x, right_stick_y, trigger_back, axis_back)


def draw_text(text, position):
    """Draw text on the screen at the given position."""
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, position)


def print_gamecontrols(screen):
    # Fill the screen with a color (background)
    screen.fill((0, 0, 0))

    # Display the status of each key
    draw_text(f"W, S:        Boom up, down", (25, -25+50))
    draw_text(f"A, D:        Boom left, right", (25, -25+100))
    draw_text(f"UP, DOWN:    Arm extend, retract", (25, -25+150))
    draw_text(f"LEFT, RIGHT: Arm lift, lower", (25, -25+200))
    draw_text(f"R, T:        Gripper rotate", (25, -25+250))
    draw_text(f"F, G:        Gripper close", (25, -25+300))

    pygame.display.flip()

    


if __name__ == "__main__":
    print("Forestry Crane Environment")

    np.set_printoptions(precision=3, suppress=True)

    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()

    if joystick_count == 0:
        print("No joystick connected.")
        gamepad_available = False

        # screen to capture keyboard input
        screen = pygame.display.set_mode((425, 320))
        pygame.display.set_caption('WASD Key Press Detection')
        print_gamecontrols(screen)
    else:
        # We assume the first joystick is the Xbox controller
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        gamepad_available = True


    print("To exit select any CV image window and press 'q'.")


    # mj_model, mj_data, env = create_robocrane_mujoco_models()
    mj_model, mj_data, env = create_forestry_crane_mujoco_models()
    renderer = mujoco.Renderer(mj_model, height=480, width=640)


    framerate = 30
    frame_time = 1.0 / framerate

    # set env visualization properties of the scene
    mj_model.vis.map.zfar = 200
    mj_model.vis.map.znear = 0.1
    mj_model.vis.map.haze = 0.05
    mj_model.vis.map.fogstart = 150
    mj_model.vis.map.fogend = 200


    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        start = time.time()
        last_frame_time = time.time()

        # camera settings (look at the crane)
        viewer.cam.azimuth = -40 
        viewer.cam.elevation = -11
        viewer.cam.distance = 16.4
        viewer.cam.lookat = [-2.98643575, -2.36544112,  0.71912036]

        # set keyframe 
        # keyframe = [-0.48,0.75,-0.3,0.4,0.4,1.3,1.5,-0.0234,0.117,0.117]
        # mj_data.qpos[0:len(keyframe)] = keyframe
        # mj_data.qacc[:] = np.zeros_like(mj_data.qacc)
        # mujoco.mj_forward(mj_model, mj_data)
        
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        
        while viewer.is_running():
            # viewer.sync()
            # if mj_data.time < mj_model.opt.timestep:
            #     print("simulation reset")
            #     mj_data.qpos[0:len(keyframe)] = keyframe
            #     mj_data.qacc[:] = np.zeros_like(mj_data.qacc)
            #     mujoco.mj_forward(mj_model, mj_data)
            #     #mujoco.mj_resetData(mj_model, mj_data)

            step_start = time.time()
            mujoco.mj_step(mj_model, mj_data)

            time_until_next_frame = frame_time - (time.time() - last_frame_time)
            if time_until_next_frame < 0:
                last_frame_time = time.time()
                viewer.sync()

                cam1_img = render_image(mj_model, mj_data, renderer, "lab/forestry_crane/camera/cam_sensor")
                cam2_img = render_image(mj_model, mj_data, renderer, "lab/forestry_crane/camera_1/cam_sensor")
                cam3_img = render_image(mj_model, mj_data, renderer, "lab/forestry_crane/camera_2/cam_sensor")
                
                cv2.imshow('cam 1', cam1_img)
                cv2.imshow('cam 2', cam2_img)
                cv2.imshow('cam 3', cam3_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                controls = user_control(gamepad_available)
                left_stick_x, left_stick_y, right_stick_x, right_stick_y, trigger_back, axis_back = controls

                scale = 0.5
                mj_data.ctrl[0] = -left_stick_x*scale
                mj_data.ctrl[1] = -left_stick_y*scale
                mj_data.ctrl[2] = -right_stick_x*scale
                mj_data.ctrl[3] = -right_stick_y*scale
                mj_data.ctrl[4] = trigger_back
                mj_data.ctrl[5] = axis_back

                print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\r".format(left_stick_x, left_stick_y, right_stick_x, right_stick_y, trigger_back, axis_back), end='')

                # print("mj_data.qpos: ", mj_data.qpos) # good for keyframe selection

            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


