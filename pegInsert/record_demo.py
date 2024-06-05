import gymnasium as gym
import sys
sys.path.insert(0, "/home/hunter/catkin_ws/src/")
import bravo_7_gym
import time
import requests
import numpy as np
import copy
import pickle as pkl
from tqdm import tqdm
import os
import datetime

from scipy.spatial.transform import Rotation as R
import numpy as np
from pyquaternion import Quaternion
from random import randint

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from bravo_7_gym.wrappers import (
#    SpacemouseIntervention,
    Quat2EulerWrapper,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper

def postNamedState(name):
    data = {"name":name, 'wait':False, 'retry':True}
    requests.post("http://127.0.0.1:5000/toNamedPose", json=data)

def getPose():
    ps = requests.post("http://127.0.0.1:5000/getstate", json={})
    ps = ps.json()
    pos = np.array(ps["pose"], dtype="float32")
    return pos


def interpolate_move(start, goal, timeout, hz):
    """Move the robot to the goal position with linear interpolation."""
    steps = int(timeout * hz)
    srpy = R.from_quat(start[3:]).as_euler("xyz")
    grpy = R.from_quat(goal[3:]).as_euler("xyz")
    xyzpath = np.linspace(start[:3], goal[:3], steps)
    rpypath = np.linspace(srpy, grpy, steps)
    path = [np.zeros((7,)) for i in range(steps)]
    for i in range(steps):
        path[i][:3] = xyzpath[i]
        path[i][3:] = R.from_euler("xyz", rpypath[i], degrees=False).as_quat()
    return path

def getLoadedPath(fp, steps=75):
    data = {"filepath":fp, 'dt':0.01, 'steps':0, 'playback':False}
    print("Waiting for traj")
    ps = requests.post("http://127.0.0.1:5000/loadTraj", json=data).json()
    raw_traj = ps['traj']
    traj = []
    for i in range(len(raw_traj)):#data['steps']):
        pt = np.array(raw_traj[i*7:i*7+7])
        traj.append(pt)
    print("return raw traj")
    return traj

if __name__ == "__main__":
    env_name = "Bravo7FixedPegInsert-v0"
    #env_name = "Bravo7_DenseRepo-v0"
    env = gym.make(env_name)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    #env = gym.wrappers.FlattenObservation(env)

    rate_hz = env.unwrapped.hz
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    demo_folder = f"recorded_paths"
    demo_path = os.path.join(file_dir, demo_folder)
    file_names = os.listdir(demo_path)
    num_demos = len(file_names)


    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"demos/fixed_peg_insert_{num_demos}_demos_{uuid}.pkl"
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")
    transitions = []
    count = 0

    print("Using Path:")
    print("\t",demo_path)
    print("Will load demo files:")
    for name in file_names:
        print("\t", name)

    pbar = tqdm(total=num_demos)
    while count < num_demos:
        curr_path = os.path.join(demo_path, file_names[count])
        print("Loading Demo:", curr_path)
        raw_poses = getLoadedPath(curr_path)#randint(10,90))
        obs, _ = env.reset()
        start_pos = raw_poses[0]
        print("Starting pose:", start_pos)
        #requests.post("http://127.0.0.1:5000/stopCC", json={})
        #time.sleep(0.5)
        #srequests.post("http://127.0.0.1:5000/pose", json={"arr":start_pos.tolist()})
        requests.post("http://127.0.0.1:5000/startCC", json={})
        time.sleep(0.5)
        print("Moved to start")
        # for each ee pose in path
        action = np.zeros((6,))
        found_goal = False
        for k in range(len(raw_poses)): #range(env.unwrapped.max_episode_length):
            print(k)
            if k < len(raw_poses):
                pose = np.array(raw_poses[k])
            else:
                pose = raw_poses[-1]
            cur_pos = np.array(env.unwrapped.currpos.copy())
            # calculate action
            action[:3] = pose[:3] - cur_pos[:3]
            action[3:] = ( R.from_quat(pose[3:]) * R.from_quat(cur_pos[3:]).inv()).as_euler("xyz")
            #print("Euler Action:", action[3:])
            # send action to env
            next_obs, rew, done, truncated, info = env.step(action)
            #print(obs.shape, type(obs))
            if info['found_goal']:
                found_goal = True

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=action,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            transitions.append(transition)

            obs = next_obs
            if done:
                print("Found goal") if found_goal else print("Failed")
                count += 1 #if found_goal else 0
                #print(
                #    f"{k}:{rew}\tGot {count} successes of {num_demos} trials."
                #)
                pbar.update(1 if found_goal else 0)
                break

    pbar.close()
    # save demos
    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {num_demos} demos to {file_path}")

    env.close()