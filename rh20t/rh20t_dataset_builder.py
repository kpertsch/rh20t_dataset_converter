from typing import Iterator, Tuple, Any

import cv2
import glob
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from rh20t.conversion_utils import MultiThreadedDatasetBuilder
from rh20t.utils import transform_timestamp_key, convert_tcp, unify_joint

TARGET_RES = (180, 320)

with open('task_description.json', 'r') as f:
    TASK_DESCRIPTION = json.load(f)
GRIPPER_THRESHOLD = 3


def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
    """Generator of examples for each split."""

    def _parse_example(episode_path):
        # load raw data --> this should change for your dataset
        scene_name, camera_1_name, camera_2_name, camera_3_name, base_path, joint_path = episode_path.split("[SPLIT]")
        task_id = int(scene_name[5:9])
        user_id = int(scene_name[15:19])
        scene_id = int(scene_name[26:30])
        cfg_id = int(scene_name[35:39])
        # 0. load metadata and audio
        with open(os.path.join(base_path, 'metadata.json'), 'r') as f:
            meta = json.load(f)
        finish_timestamp = meta['finish_time']
        rating = meta['rating']
        # audio_file = os.path.join(base_path, 'audio_mixed')
        # audio_file = os.path.join(audio_file, os.listdir(audio_file)[0])
        # 1. load color data and timestamps
        images = []
        cam_sns = []
        for camera_name in [camera_1_name, camera_2_name, camera_3_name]:
            color_path = os.path.join(base_path, camera_name)
            cam_sn = camera_name[4:]
            cam_sns.append(cam_sn)
            timestamps = np.load(os.path.join(color_path, 'timestamps.npy'), allow_pickle=True).item()
            cap = cv2.VideoCapture(os.path.join(color_path, 'color.mp4'))
            colors = {}
            cnt = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    # height, width, _ = frame.shape
                    frame = cv2.resize(frame, TARGET_RES)
                    frame = np.array(frame).astype(np.uint8)
                    frame = frame[..., ::-1]    # flip channel order bc OpenCV, oh well
                    colors[timestamps['color'][cnt]] = frame
                    cnt += 1
                else:
                    break
            cap.release()
            images.append(colors)
        # 2. load depth data (if any)
        # if os.path.exists(os.path.join(depth_path, 'depth.mp4')):
        #     cap = cv2.VideoCapture(os.path.join(depth_path, 'depth.mp4'))
        #     depths = {}
        #     cnt = 0
        #     while True:
        #         ret, frame = cap.read()
        #         if ret:
        #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #             gray1 = np.array(gray[:height, :]).astype(np.uint16)
        #             gray2 = np.array(gray[height:, :]).astype(np.uint16)
        #             depths[timestamps['depth'][cnt]] = np.array(gray2 * 256 + gray1).astype(np.uint16)
        #             cnt += 1
        #         else:
        #             break
        # else:
        #     depths = None

        # 3. load transformed data
        # 3.1 tcp w.r.t. camera
        tcps = np.load(os.path.join(base_path, 'transformed', 'tcp.npy'), allow_pickle=True).item()
        tcps = transform_timestamp_key(tcps)

        # 3.2 tcp w.r.t. base
        tcps_base = transform_timestamp_key(
            np.load(os.path.join(base_path, 'transformed', 'tcp_base.npy'), allow_pickle=True).item())

        # 3.3 force/torque w.r.t. camera
        # fts = np.load(os.path.join(base_path, 'transformed', 'force_torque.npy'), allow_pickle=True).item()
        # fts = transform_timestamp_key(fts)

        # 3.4 force/torque w.r.t. base
        if os.path.exists(os.path.join(base_path, 'transformed', 'force_torque_base.npy')):
            fts_base = transform_timestamp_key(
                np.load(os.path.join(base_path, 'transformed', 'force_torque_base.npy'), allow_pickle=True).item())
        else:
            fts_base = None

        # 3.5 gripper
        grippers = np.load(os.path.join(base_path, 'transformed', 'gripper.npy'), allow_pickle=True).item()

        # 4. load joint data (if any)
        if os.path.exists(os.path.join(joint_path, 'transformed', 'joint.npy')):
            joints = np.load(os.path.join(joint_path, 'transformed', 'joint.npy'), allow_pickle=True).item()
        else:
            joints = None

        # 5. language instruction
        language_instruction = TASK_DESCRIPTION[scene_name[:9]]["task_description_english"]

        # 6. timestamps: find union over cameras
        all_timestamps = [list(imgs.keys()) for imgs in images]
        intersect_timestamps = set(all_timestamps[0]).intersect(*all_timestamps[1:])
        timestamp_base = sorted(list(intersect_timestamps))

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        last_gripper_action = 1
        for i, ts in enumerate(timestamp_base):
            if i == len(timestamp_base) - 1:
                break
            next_ts = timestamp_base[i + 1]
            is_last = (i == len(timestamp_base) - 2) or (ts < finish_timestamp and next_ts >= finish_timestamp)

            try:
                tcp_0 = convert_tcp(tcps[cam_sns[0]][ts]['tcp'])
                tcp_1 = convert_tcp(tcps[cam_sns[1]][ts]['tcp'])
                tcp_2 = convert_tcp(tcps[cam_sns[2]][ts]['tcp'])
                tcp_base = convert_tcp(tcps_base[cam_sn][ts]['tcp'])
                tcp_base_action = convert_tcp(tcps_base[cam_sn][next_ts]['tcp']) - tcp_base
            except Exception:
                return None
            try:
                joint, joint_vel = unify_joint(joints[cam_sns[0]][ts], cfg_id)
            except Exception:
                joint = np.zeros(7).astype(np.float32)
                joint_vel = np.zeros(7).astype(np.float32)
            # try:
            #     ft_robot = np.array(tcps[cam_sn][ts]['robot_ft']).astype(np.float32)
            # except Exception:
            #     ft_robot = np.zeros(6).astype(np.float32)
            try:
                ft_robot_base = np.array(tcps_base[cam_sns[0]][ts]['robot_ft']).astype(np.float32)
            except Exception:
                ft_robot_base = np.zeros(6).astype(np.float32)
            # try:
            #     ft_raw = np.array(fts[cam_sn][ts]['raw']).astype(np.float32)
            # except Exception:
            #     ft_raw = np.zeros(6).astype(np.float32)
            # try:
            #     ft_zeroed = np.array(fts[cam_sn][ts]['zeroe']).astype(np.float32)
            # except Exception:
            #     ft_zeroed = np.zeros(6).astype(np.float32)
            try:
                ft_raw_base = np.array(fts_base[cam_sns[0]][ts]['raw']).astype(np.float32)
            except Exception:
                ft_raw_base = np.zeros(6).astype(np.float32)
            try:
                ft_zeroed_base = np.array(fts_base[cam_sns[0]][ts]['zeroed']).astype(np.float32)
            except Exception:
                ft_zeroed_base = np.zeros(6).astype(np.float32)
            try:
                gripper_width = grippers[cam_sns[0]][ts]['gripper_info'][0]
                gripper_next_width = grippers[cam_sns[0]][next_ts]['gripper_info'][0]
            except Exception:
                return None
            if np.abs(gripper_width - gripper_next_width) >= GRIPPER_THRESHOLD:
                if gripper_next_width < gripper_width:
                    gripper_action = 0
                else:
                    gripper_action = 1
            else:
                gripper_action = last_gripper_action
            last_gripper_action = gripper_action
            # try:
            #     tactile = fetch_tactile(tactiles, ts)
            # except Exception:
            #     tactile = np.zeros(96).astype(np.int32)
            episode.append({
                'observation': {
                    'image_0': images[0][ts],
                    'image_1': images[1][ts],
                    'image_2': images[2][ts],
                    # 'depth': depth,
                    'tcp_0': tcp_0,
                    'tcp_1': tcp_1,
                    'tcp_2': tcp_2,
                    'tcp_base': tcp_base,
                    'joint': joint,
                    'joint_vel': joint_vel,
                    # 'ft_robot': ft_robot,
                    'ft_robot_base': ft_robot_base,
                    # 'ft_raw': ft_raw,
                    # 'ft_zeroed': ft_zeroed,
                    'ft_raw_base': ft_raw_base,
                    'ft_zeroed_base': ft_zeroed_base,
                    'gripper_width': gripper_width,
                    # 'tactile': tactile,
                    'timestamp': ts
                },
                'action': {
                    # 'tcp': tcp_action,
                    'tcp_base': tcp_base_action,
                    'gripper': gripper_action
                },
                'discount': 1.0,
                'reward': float(rating >= 2 and is_last),
                'is_first': i == 0,
                'is_last': is_last,
                'is_terminal': is_last,
                'language_instruction': language_instruction,
                # 'language_embedding': language_embedding,
            })

        # invalid episode check
        if episode:
            return None

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': base_path,
                'camera_ids': str(cam_sns),
                'task_id': task_id,
                'user_id': user_id,
                'scene_id': scene_id,
                'cfg_id': cfg_id,
                # 'audio': audio_file,
                'finish_timestamp': finish_timestamp,
                'rating': rating
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)


class Rh20tDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 10             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 100  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_0': tfds.features.Image(
                            shape=(TARGET_RES[0], TARGET_RES[1], 3),  # (720, 1280) for original, (360, 640) for resized
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Camera 0 RGB observation.'
                        ),
                        'image_1': tfds.features.Image(
                            shape=(TARGET_RES[0], TARGET_RES[1], 3),  # (720, 1280) for original, (360, 640) for resized
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Camera 1 RGB observation.'
                        ),
                        'image_2': tfds.features.Image(
                            shape=(TARGET_RES[0], TARGET_RES[1], 3),  # (720, 1280) for original, (360, 640) for resized
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Camera 2 RGB observation.'
                        ),
                        # 'depth': tfds.features.Image(
                        #     shape=(None, None, 1),  # (720, 1280) for original, (360, 640) for resized
                        #     dtype=np.uint16,
                        #     encoding_format='png',
                        #     doc='Camera depth observation (in mm).'
                        # ),
                        'tcp_0': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot tcp pose [3x xyz + 3x rpy] in the camera 0 coordinate.'
                        ),
                        'tcp_1': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot tcp pose [3x xyz + 3x rpy] in the camera 1 coordinate.'
                        ),
                        'tcp_2': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot tcp pose [3x xyz + 3x rpy] in the camera 2 coordinate.'
                        ),
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
                        # 'ft_robot': tfds.features.Tensor(
                        #     shape=(6,),
                        #     dtype=np.float32,
                        #     doc='Force/torque [3x force + 3x torque] value received from the robot in the camera coordinate.'
                        # ),
                        # 'ft_raw': tfds.features.Tensor(
                        #     shape=(6,),
                        #     dtype=np.float32,
                        #     doc='Force/torque [3x force + 3x torque] raw value received from the sensor in the camera coordinate.',
                        # ),
                        # 'ft_zeroed': tfds.features.Tensor(
                        #     shape=(6,),
                        #     dtype=np.float32,
                        #     doc='Force/torque [3x force + 3x torque] zeroed value (after taring) received from the sensor in the camera coordinate.'
                        # ),
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
                        # 'tactile': tfds.features.Tensor(
                        #     shape=(96,),
                        #     dtype=np.int32,
                        #     doc='Tactile information (only available in RH20T cfg7).'
                        # ),
                        'timestamp': tfds.features.Scalar(
                            dtype=np.int64,
                            doc='Timestamp of this data record.'
                        )
                    }),
                    'action': tfds.features.FeaturesDict({
                        # 'tcp': tfds.features.Tensor(
                        #     shape=(6,),
                        #     dtype=np.float32,
                        #     doc='Tcp action (in camera coordinate).'
                        # ),
                        'tcp_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Tcp action (in robot base coordinate).'
                        ),
                        'gripper': tfds.features.Scalar(
                            dtype=np.int64,
                            doc='Gripper action.'
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
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'camera_id': tfds.features.Text(
                        doc='Camera serial number.'
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
                    # 'audio': tfds.features.Audio(
                    #     file_format='wav',
                    #     doc='Audio.'
                    # ),
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

    def _generate_paths(self, path, joint_path=None):
        """Generator of paths: RH20T format configuration/scene/camera"""
        paths = []
        for conf in sorted(os.listdir(path)):
            for scene in sorted(os.listdir(os.path.join(path, conf))):
                if '_human' in scene or 'calib' in scene:
                    continue
                if not os.path.exists(os.path.join(path, conf, scene, 'transformed')) or \
                   not os.path.exists(os.path.join(path, conf, scene, 'transformed', 'tcp.npy')) or \
                   not os.path.exists(os.path.join(path, conf, scene, 'transformed', 'tcp_base.npy')) or \
                   not os.path.exists(os.path.join(path, conf, scene, 'transformed', 'gripper.npy')) or \
                   not os.path.exists(os.path.join(path, conf, scene, 'transformed', 'force_torque.npy')):
                    continue
                try:
                    task_id = int(scene[5:9])
                    user_id = int(scene[15:19])
                    scene_id = int(scene[26:30])
                    cfg_id = int(scene[35:39])
                except Exception:
                    continue
                cams = glob.glob(os.path.join(path, conf, scene, "cam_*"))
                cam_paths = []
                for camera in sorted(cams):
                    if not os.path.exists(os.path.join(path, conf, scene, camera, 'color.mp4')) or \
                       not os.path.exists(os.path.join(path, conf, scene, camera, 'timestamps.npy')):
                        continue
                    cam_paths.append(os.path.join(path, conf, scene, camera))
                if len(cam_paths) < 3:
                    continue
                paths.append(
                    scene + "[SPLIT]" +
                    cam_paths[0] + "[SPLIT]" +
                    cam_paths[1] + "[SPLIT]" +
                    cam_paths[2] + "[SPLIT]" +
                    os.path.join(path, conf, scene) + "[SPLIT]" +
                    # ("" if depth_path is None or not os.path.exists(
                    #     os.path.join(depth_path, conf, scene, camera)) else os.path.join(depth_path, conf,
                    #                                                                      scene,
                    #                                                                      camera)) + "[SPLIT]" +
                    ("" if joint_path is None or not os.path.exists(
                        os.path.join(joint_path, conf, scene)) else os.path.join(joint_path, conf, scene))
                )
        return paths

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            'train': self._generate_paths(
                path='/nfs/kun2/datasets/rh20t',
                joint_path='/nfs/kun2/datasets/rh20t'
            ),
        }
