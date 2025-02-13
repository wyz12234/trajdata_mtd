from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from tqdm import trange

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.data_structures.scene_metadata import Scene
from trajdata.data_structures.state import StateArray
from trajdata.simulation import SimulationScene, sim_metrics, sim_stats, sim_vis
from trajdata.visualization.vis import plot_agent_batch

import os
import json


def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_trainval-val"],
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 50.0),
        # incl_map=True,
        # map_params={
        #     "px_per_m": 2,
        #     "map_size_px": 224,
        #     "offset_frac_xy": (0.0, 0.0),
        # },
        verbose=True,
        # desired_dt=0.1,
        num_workers=4,
        data_dirs={  # Remember to change this to match your filesystem!
            "nusc_trainval" : "~/behavior-generation-dataset/nuscenes",
        },
    )

    ade = sim_metrics.ADE()
    fde = sim_metrics.FDE()

    sim_env_name = "nusc_mini_sim"
    all_sim_scenes: List[Scene] = list()
    desired_scene: Scene
    sim_stats_dict = dict()
    for idx, desired_scene in enumerate(dataset.scenes()):
        sim_scene: SimulationScene = SimulationScene(
            env_name=sim_env_name,
            scene_name=f"sim_scene-{idx:04d}",
            scene=desired_scene,
            dataset=dataset,
            init_timestep=0,
            freeze_agents=True,
        )

        vel_hist = sim_stats.VelocityHistogram(bins=np.linspace(0, 30, 21))
        lon_acc_hist = sim_stats.LongitudinalAccHistogram(bins=np.linspace(0, 10, 21))
        lat_acc_hist = sim_stats.LateralAccHistogram(bins=np.linspace(0, 10, 21))
        jerk_hist = sim_stats.JerkHistogram(
            bins=np.linspace(0, 20, 21), dt=sim_scene.scene.dt
        )

        obs: AgentBatch = sim_scene.reset()
        for t in trange(0, sim_scene.scene.length_timesteps):
            new_xyzh_dict: Dict[str, StateArray] = dict()
            for idx, agent_name in enumerate(obs.agent_name):
                curr_yaw = obs.curr_agent_state[idx].heading.item()
                curr_pos = obs.curr_agent_state[idx].position.numpy()
                world_from_agent = np.array(
                    [
                        [np.cos(curr_yaw), np.sin(curr_yaw)],
                        [-np.sin(curr_yaw), np.cos(curr_yaw)],
                    ]
                )
                next_state = np.zeros((4,))
                if obs.agent_fut_len[idx] < 1:
                    next_state[:2] = curr_pos
                    yaw_ac = 0
                else:
                    next_state[:2] = (
                        obs.agent_fut[idx, 0].position.numpy() @ world_from_agent
                        + curr_pos
                    )
                    yaw_ac = obs.agent_fut[idx, 0].heading.item()

                next_state[-1] = curr_yaw + yaw_ac
                new_xyzh_dict[agent_name] = StateArray.from_array(next_state, "x,y,z,h")

            obs = sim_scene.step(new_xyzh_dict)
            metrics: Dict[str, Dict[str, float]] = sim_scene.get_metrics([ade, fde])
            # print(metrics)

        stats: Dict[
            str, Dict[str, Tuple[np.ndarray, np.ndarray]]
        ] = sim_scene.get_stats([vel_hist, lon_acc_hist, lat_acc_hist, jerk_hist])
        stats_gt = {
            'vel_hist': stats['vel_hist']['sim'],
            'lon_acc_hist': stats['lon_acc_hist']['sim'],
            'lat_acc_hist': stats['lat_acc_hist']['sim'],
            'jerk_hist': stats['jerk_hist']['sim'],
        }
        for k in stats_gt:
            if k not in sim_stats_dict:
                sim_stats_dict[k] = stats_gt[k][0]
            else:
                sim_stats_dict[k] += stats_gt[k][0]
        ticks = {
            'vel_hist': stats['vel_hist']['gt'][1],
            'lon_acc_hist': stats['lon_acc_hist']['gt'][1],
            'lat_acc_hist': stats['lat_acc_hist']['gt'][1],
            'jerk_hist': stats['jerk_hist']['gt'][1],
        }
        # sim_vis.plot_sim_stats(stats)

        # plot_agent_batch(obs, 0, show=False, close=False)
        # plot_agent_batch(obs, 1, show=False, close=False)
        # plot_agent_batch(obs, 2, show=False, close=False)
        # plot_agent_batch(obs, 3, show=True, close=True)

        sim_scene.finalize()
        sim_scene.save()

        all_sim_scenes.append(sim_scene.scene)
    output_file = os.path.join("hist_stats_gt.json")
    for k in sim_stats_dict:
        sim_stats_dict[k] = (sim_stats_dict[k] / len(all_sim_scenes)).tolist()
    for k in ticks:
        ticks[k] = ticks[k].tolist()

    json.dump({"stats": sim_stats_dict, "ticks": ticks}, open(output_file, "w+"), indent=4)
    # dataset.env_cache.save_env_scenes_list(sim_env_name, all_sim_scenes)


if __name__ == "__main__":
    main()
