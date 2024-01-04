from typing import Iterator, Tuple, Any

import cv2
import os
import glob
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from rh20t.conversion_utils import MultiThreadedDatasetBuilder
from rh20t.utils import transform_timestamp_key, convert_tcp, unify_joint
from rh20t.cam_definitions import CFG_TO_CAM

TARGET_RES = (180, 320)     # original res: (720, 1280)

with open('task_description.json', 'r') as f:
    TASK_DESCRIPTION = json.load(f)     # 120 unique tasks
GRIPPER_THRESHOLD = 3
MAX_TIME_DIFF_SYNC = 100        # max time diff in ms for cam frames to count as synced

CAMS_TO_CONVERT = list(CFG_TO_CAM['cfg1'].keys())       # which of the cameras should be converted

DATA_PATH = "/nfs/kun2/datasets/rh20t"
JOINT_INFO_PATH = "/nfs/kun2/datasets/rh20t/RH20T_joint"


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Generator of examples for each split."""

    def _parse_example(episode_path):
        # load raw data --> this should change for your dataset
        scene_name = os.path.basename(os.path.normpath(episode_path))
        cfg_name = episode_path.split('/')[-2]
        task_id = int(scene_name[5:9])
        user_id = int(scene_name[15:19])
        scene_id = int(scene_name[26:30])
        cfg_id = int(scene_name[35:39])
        # 0. load metadata
        try:
            with open(os.path.join(episode_path, 'metadata.json'), 'r') as f:
                meta = json.load(f)
            finish_timestamp = meta['finish_time']
            rating = meta['rating']
        except Exception:
            print(f"Failed to load metadata for {episode_path}")
            return None

        # 1. load camera data and timestamps
        images = {}
        timestamps = {}
        try:
            for camera_name in CFG_TO_CAM[f"cfg{cfg_id}"]:
                cam_path = os.path.join(episode_path, f"cam_{CFG_TO_CAM[f'cfg{cfg_id}'][camera_name]}")
                timestamps[camera_name] = np.array(
                    np.load(os.path.join(cam_path, 'timestamps.npy'), allow_pickle=True).item()['color']
                )
                cap = cv2.VideoCapture(os.path.join(cam_path, 'color.mp4'))
                cnt = 0
                images[camera_name] = {}
                while True:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (TARGET_RES[1], TARGET_RES[0]), interpolation=cv2.INTER_AREA)
                        frame = np.array(frame).astype(np.uint8)
                        frame = frame[..., ::-1]    # flip channel order bc OpenCV, oh well
                        images[camera_name][timestamps[camera_name][cnt]] = frame
                        cnt += 1
                    else:
                        break
                cap.release()
        except Exception:
            print(f"Failed image extraction for {episode_path}")
            return None

        # 3. load transformed data
        # 3.1 tcp w.r.t. camera
        try:
            tcps = np.load(os.path.join(episode_path, 'transformed', 'tcp.npy'), allow_pickle=True).item()
            tcps = transform_timestamp_key(tcps)

            # 3.2 tcp w.r.t. base
            tcps_base = transform_timestamp_key(
                np.load(os.path.join(episode_path, 'transformed', 'tcp_base.npy'), allow_pickle=True).item())
        except Exception:
                print(f"Failed to load TCP for {episode_path}")
                return None

        # 3.4 force/torque w.r.t. base
        if os.path.exists(os.path.join(episode_path, 'transformed', 'force_torque_base.npy')):
            try:
                fts_base = transform_timestamp_key(
                    np.load(os.path.join(episode_path, 'transformed', 'force_torque_base.npy'), allow_pickle=True).item())
            except Exception:
                print(f"Failed to load force torque for {episode_path}")
                return None
        else:
            fts_base = None

        # 3.5 gripper
        try:
            grippers = np.load(os.path.join(episode_path, 'transformed', 'gripper.npy'), allow_pickle=True).item()
        except Exception:
            print(f"Failed to load gripper info for {episode_path}")
            return None

        # 4. load joint data (if any)
        if os.path.exists(os.path.join(JOINT_INFO_PATH, cfg_name, scene_name, 'transformed', 'joint.npy')):
            try:
                joints = np.load(
                    os.path.join(JOINT_INFO_PATH, cfg_name, scene_name, 'transformed', 'joint.npy'),
                    allow_pickle=True).item()
            except Exception:
                print(f"Failed to load joint info for {episode_path}")
                return None
        else:
            joints = None

        # 5. language instruction
        try:
            language_instruction = TASK_DESCRIPTION[scene_name[:9]]["task_description_english"]
        except Exception:
            print(f"Failed language instruction extraction for {episode_path}")
            return None

        # 6. timestamps: find time aligned camera timestamps
        sync_timestamps = {cam: [] for cam in CAMS_TO_CONVERT}
        for timestamp in timestamps[CAMS_TO_CONVERT[0]]:    # use first cam's time stamps as base
            if all([min(np.abs(timestamps[cam] - timestamp)) < MAX_TIME_DIFF_SYNC for cam in CAMS_TO_CONVERT]):
                for cam in CAMS_TO_CONVERT:
                    sync_step = np.argmin(np.abs(timestamps[cam] - timestamp))
                    sync_timestamps[cam].append(timestamps[cam][sync_step])

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        last_gripper_action = 1
        base_cam_sns = CFG_TO_CAM[f"cfg{cfg_id}"][CAMS_TO_CONVERT[0]]
        for i, ts in enumerate(sync_timestamps[CAMS_TO_CONVERT[0]]):
            if i == len(sync_timestamps[CAMS_TO_CONVERT[0]]) - 1:
                break
            next_ts = sync_timestamps[CAMS_TO_CONVERT[0]][i + 1]
            is_last = (i == len(sync_timestamps[CAMS_TO_CONVERT[0]]) - 2) or \
                      (ts < finish_timestamp and next_ts >= finish_timestamp)

            try:
                tcp_per_cam = {}
                for cam in CAMS_TO_CONVERT:
                    cam_sns = CFG_TO_CAM[f"cfg{cfg_id}"][cam]
                    cam_ts = sync_timestamps[cam][i]
                    tcp_per_cam[cam] = convert_tcp(tcps[cam_sns][cam_ts]['tcp'])
                tcp_base = convert_tcp(tcps_base[base_cam_sns][ts]['tcp'])
                tcp_base_action = convert_tcp(tcps_base[base_cam_sns][next_ts]['tcp']) - tcp_base
            except Exception:
                print(f"Failed TCP extraction for {episode_path}")
                return None
            try:
                joint, joint_vel = unify_joint(joints[base_cam_sns][ts], cfg_id)
            except Exception:
                joint = np.zeros(7).astype(np.float32)
                joint_vel = np.zeros(7).astype(np.float32)
            try:
                ft_robot_base = np.array(tcps_base[base_cam_sns][ts]['robot_ft']).astype(np.float32)
            except Exception:
                print("Failed force-torque extraction")
                ft_robot_base = np.zeros(6).astype(np.float32)
            try:
                ft_raw_base = np.array(fts_base[base_cam_sns][ts]['raw']).astype(np.float32)
            except Exception:
                print("Failed force-torque raw extraction")
                ft_raw_base = np.zeros(6).astype(np.float32)
            try:
                ft_zeroed_base = np.array(fts_base[base_cam_sns][ts]['zeroed']).astype(np.float32)
            except Exception:
                print("Failed force-torque zeroed extraction")
                ft_zeroed_base = np.zeros(6).astype(np.float32)
            try:
                gripper_width = grippers[base_cam_sns][ts]['gripper_info'][0]
                gripper_next_width = grippers[base_cam_sns][next_ts]['gripper_info'][0]
            except Exception:
                print(f"Failed gripper_width extraction for {episode_path}")
                return None
            if np.abs(gripper_width - gripper_next_width) >= GRIPPER_THRESHOLD:
                if gripper_next_width < gripper_width:
                    gripper_action = 0
                else:
                    gripper_action = 1
            else:
                gripper_action = last_gripper_action
            last_gripper_action = gripper_action

            try:
                episode.append({
                    'observation': {
                        **{
                            f"image_{cam}": images[cam][sync_timestamps[cam][i]]
                            for cam in CAMS_TO_CONVERT
                        },
                        **{
                            f"tcp_{cam}": tcp_per_cam[cam]
                            for cam in CAMS_TO_CONVERT
                        },
                        'tcp_base': tcp_base,
                        'joint': joint,
                        'joint_vel': joint_vel,
                        'ft_robot_base': ft_robot_base,
                        'ft_raw_base': ft_raw_base,
                        'ft_zeroed_base': ft_zeroed_base,
                        'gripper_width': gripper_width,
                        'timestamp': ts
                    },
                    'action': {
                        'tcp_base': tcp_base_action,
                        'gripper': gripper_action
                    },
                    'discount': 1.0,
                    'reward': float(rating >= 2 and is_last),
                    'is_first': i == 0,
                    'is_last': is_last,
                    'is_terminal': is_last,
                    'language_instruction': language_instruction,
                })
            except Exception:
                print(f"Failed episode assembly for {episode_path}")
                return None

        # invalid episode check
        if len(episode) < 10:
            print(f"Too few steps in {episode_path}")
            return None

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path,
                'task_id': task_id,
                'user_id': user_id,
                'scene_id': scene_id,
                'cfg_id': cfg_id,
                'finish_timestamp': finish_timestamp,
                'rating': rating
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)


class Rh20t(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 400  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        **{
                            f'image_{k}': tfds.features.Image(
                                shape=(TARGET_RES[0], TARGET_RES[1], 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc=f'RGB observation for {k} camera.'
                            )
                            for k in CAMS_TO_CONVERT
                        },
                        **{
                            f'tcp_{k}': tfds.features.Tensor(
                                shape=(6,),
                                dtype=np.float32,
                                doc=f'Robot tcp pose [3x xyz + 3x rpy] in coordinates of camera {k}.'
                            )
                            for k in CAMS_TO_CONVERT
                        },
                        'tcp_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot tcp pose [3x xyz + 3x rpy] in the robot base coordinate.'
                        ),
                        'joint': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint pose [7x position for 7-DoF settings, and 6x for 6-DoF settings].'
                        ),
                        'joint_vel': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint pose [7x position for 7-DoF settings, and 6x for 6-DoF settings];'
                                'all 0 means the configuration does not provide joint velocity data.'
                        ),
                        'ft_robot_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Force/torque [3x force + 3x torque] value received from the robot in the robot base coordinate.'
                        ),
                        'ft_raw_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Force/torque [3x force + 3x torque] raw value received from the sensor in the robot base coordinate.'
                        ),
                        'ft_zeroed_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Force/torque [3x force + 3x torque] zeroed value (after taring) received from the sensor in the robot base coordinate.'
                        ),
                        'gripper_width': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Gripper width in mm.'
                        ),
                        'timestamp': tfds.features.Scalar(
                            dtype=np.int64,
                            doc='Timestamp of this data record.'
                        )
                    }),
                    'action': tfds.features.FeaturesDict({
                        'tcp_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Tcp action (in robot base coordinate).'
                        ),
                        'gripper': tfds.features.Scalar(
                            dtype=np.int64,
                            doc='Gripper action. 0 = closed, 1 = open'
                        )
                    }),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'task_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Task ID.'
                    ),
                    'user_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='User ID.'
                    ),
                    'scene_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Scene ID.'
                    ),
                    'cfg_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Configuration ID.'
                    ),
                    'finish_timestamp': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Finish timestamp.'
                    ),
                    'rating': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='User rating (0: robot entered the emergency state; 1: task failed; 2-9: user evaluation of manipulation quality).'
                    )
                }),
            }))

    def _generate_paths(self, path):
        """Generator of paths: RH20T format configuration/scene/camera"""
        paths = []
        failed_on_transform = 0
        failed_on_cams = 0
        failed_on_scene = 0
        from collections import defaultdict
        cam_fail_counter = defaultdict(lambda: defaultdict(lambda: 0))
        for conf in sorted(os.listdir(path)):
            if 'tar.gz' in conf or '.sh' in conf: continue
            for scene in sorted(os.listdir(os.path.join(path, conf))):
                if '_human' in scene or 'calib' in scene:
                    continue
                if not os.path.exists(os.path.join(path, conf, scene, 'transformed')) or \
                   not os.path.exists(os.path.join(path, conf, scene, 'transformed', 'tcp.npy')) or \
                   not os.path.exists(os.path.join(path, conf, scene, 'transformed', 'tcp_base.npy')) or \
                   not os.path.exists(os.path.join(path, conf, scene, 'transformed', 'gripper.npy')) or \
                   not os.path.exists(os.path.join(path, conf, scene, 'transformed', 'force_torque.npy')):
                       failed_on_transform += 1
                       continue
                try:
                    task_id = int(scene[5:9])
                    user_id = int(scene[15:19])
                    scene_id = int(scene[26:30])
                    cfg_id = int(scene[35:39])
                except Exception:
                    failed_on_scene += 1
                    continue
                cams = CFG_TO_CAM[f"cfg{cfg_id}"].values()
                cams_in_dir = os.listdir(os.path.join(path, conf, scene))
                valid = True
                for cam in cams:
                    if not f"cam_{cam}" in cams_in_dir or \
                      not os.path.exists(os.path.join(path, conf, scene, f"cam_{cam}", 'color.mp4')) or \
                      not os.path.exists(os.path.join(path, conf, scene, f"cam_{cam}", 'timestamps.npy')):
                        valid = False
                        cam_name = list(CFG_TO_CAM[f"cfg{cfg_id}"].keys())[list(CFG_TO_CAM[f"cfg{cfg_id}"].values()).index(cam)]
                        cam_fail_counter[cfg_id][cam_name] += 1
                if not valid:
                    failed_on_cams += 1
                    continue

                # passed all tests, append path
                paths.append(os.path.join(path, conf, scene))
        print("Missing transform: ", failed_on_transform)
        print("Missing cameras: ", failed_on_cams)
        print("Can't parse scene: ", failed_on_scene)
        summed_cams = defaultdict(lambda: 0)
        for cfg in cam_fail_counter:
            for cam in cam_fail_counter[cfg]:
                summed_cams[cam] += cam_fail_counter[cfg][cam]
        print("Missing the following cameras: ", dict(summed_cams))
        return paths

    def _split_paths(self):
        """Define filepaths for data splits."""
        paths = self._generate_paths(path=DATA_PATH)
        print(f"Converting {len(paths)} episodes.")
        return {
            'train': paths
        }
