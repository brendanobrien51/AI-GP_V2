"""Core DirectRLEnv for autonomous drone racing in Isaac Lab.

This environment manages:
    - Spawning and resetting the Neros drone + gate assets
    - Applying CTBR actions to the drone physics
    - Computing observations (actor vector + image, critic privileged)
    - Computing rewards and tracking gate-pass progress
    - Termination / truncation logic
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from aigp.envs.actions import clamp_actions, ctbr_to_motor_forces, scale_ctbr_actions
from aigp.envs.racing_env_cfg import RacingEnvCfg
from aigp.envs.rewards import compute_total_reward
from aigp.envs.terminations import check_collision, check_out_of_bounds, check_timeout
from aigp.track.track_generator import generate_zigzag, generate_split_s, generate_circular
from aigp.utils.math_utils import quat_rotate_inverse, quat_to_gravity_body, world_to_body


_TRACK_GENERATORS = {
    "zigzag": generate_zigzag,
    "split_s": generate_split_s,
    "circular": generate_circular,
}


class RacingEnv(DirectRLEnv):
    """Isaac Lab DirectRLEnv for drone gate racing.

    Observation dict keys:
        ``"policy"``: (N, 13) actor vector observations
        ``"critic"``: (N, 31) privileged critic observations
        ``"image"``:  (N, 3, 80, 80) RGB from TiledCamera
    """

    cfg: RacingEnvCfg

    def __init__(self, cfg: RacingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Per-env state tracking
        self._step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._prev_action = torch.zeros(self.num_envs, 4, device=self.device)
        self._current_gate_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._prev_dist_to_gate = torch.zeros(self.num_envs, device=self.device)

        # Gate positions: (num_envs, max_gates, 3)
        self._gate_positions = torch.zeros(
            self.num_envs, self.cfg.max_num_gates, 3, device=self.device
        )
        self._active_num_gates = torch.full(
            (self.num_envs,), self.cfg.num_gates, device=self.device, dtype=torch.long
        )
        self._gates_passed = torch.zeros(
            self.num_envs, self.cfg.max_num_gates, device=self.device, dtype=torch.bool
        )

        # Seed counter for track generation reproducibility
        self._track_seed = 0

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Spawn drone, gates, camera, and ground plane via scene config."""
        # Scene is configured declaratively in RacingSceneCfg.
        # InteractiveScene handles spawning based on the config.
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Cache handles for fast access
        self._drone = self.scene["drone"]
        try:
            self._camera = self.scene["camera"]
        except KeyError:
            self._camera = None

        # Build list of gate handles
        self._gate_handles = []
        for i in range(8):
            key = f"gate_{i}"
            try:
                self._gate_handles.append(self.scene[key])
            except KeyError:
                break

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Scale and cache raw policy actions before physics stepping."""
        self._raw_actions = clamp_actions(actions)
        self._scaled_actions = scale_ctbr_actions(self._raw_actions)

    def _apply_action(self):
        """Convert CTBR to motor forces/torques and apply to the drone."""
        forces, torques = ctbr_to_motor_forces(self._scaled_actions)

        # Apply motor forces to rotor joints
        self._drone.set_joint_effort_target(forces)

        # Apply body torques as external forces
        self._drone.set_external_force_and_torque(
            forces=torch.zeros(self.num_envs, 1, 3, device=self.device),
            torques=torques.unsqueeze(1),
            body_ids=[0],
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Build actor, critic, and image observations.

        Returns:
            Dict with keys "policy" (N, 13), "critic" (N, 31),
            and "image" (N, 3, H, W).
        """
        root_state = self._drone.data.root_state_w  # (N, 13)
        pos = root_state[:, :3]       # (N, 3)
        quat = root_state[:, 3:7]     # (N, 4) wxyz
        vel = root_state[:, 7:10]     # (N, 3)
        ang_vel = root_state[:, 10:13]  # (N, 3)

        # Angular velocity in body frame
        ang_vel_body = quat_rotate_inverse(quat, ang_vel)  # (N, 3)

        # Gravity in body frame
        gravity_body = quat_to_gravity_body(quat)  # (N, 3)

        # Relative position of next gate in body frame
        next_gate_idx = self._current_gate_idx.clamp(max=self.cfg.max_num_gates - 1)
        next_gate_pos = self._gate_positions[
            torch.arange(self.num_envs, device=self.device), next_gate_idx
        ]  # (N, 3)
        rel_gate_body = world_to_body(next_gate_pos, pos, quat)  # (N, 3)

        # Actor observation: 13D
        actor_obs = torch.cat([
            ang_vel_body,          # 3
            gravity_body,          # 3
            rel_gate_body,         # 3
            self._prev_action,     # 4
        ], dim=-1)  # (N, 13)

        # Critic observation: 31D (privileged)
        # Include up to 4 future gate positions in world frame
        future_gate_flat = self._gate_positions[:, :4].reshape(self.num_envs, -1)  # (N, 12)
        critic_obs = torch.cat([
            actor_obs,             # 13
            pos,                   # 3
            vel,                   # 3
            future_gate_flat,      # 12
        ], dim=-1)  # (N, 31)

        obs = {
            "policy": actor_obs,
            "critic": critic_obs,
        }

        # Image observations from TiledCamera (if available)
        cam_data = self._camera.data.output.get("rgb") if self._camera is not None else None
        if cam_data is not None:
            # TiledCamera returns (N, H, W, 3) uint8 -> convert to float [0, 1]
            images = cam_data.float() / 255.0
            # Rearrange to (N, 3, H, W) for CNN
            images = images.permute(0, 3, 1, 2)
            obs["image"] = images

        return obs

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        """Compute per-env reward for the current step."""
        pos = self._drone.data.root_state_w[:, :3]

        # Distance to current target gate
        next_idx = self._current_gate_idx.clamp(max=self.cfg.max_num_gates - 1)
        next_gate_pos = self._gate_positions[
            torch.arange(self.num_envs, device=self.device), next_idx
        ]
        dist_to_gate = torch.norm(pos - next_gate_pos, dim=-1)

        # Check gate passing (drone crosses gate plane)
        passed = dist_to_gate < 1.5  # within 1.5m of gate center
        just_passed = passed & ~self._gates_passed[
            torch.arange(self.num_envs, device=self.device), next_idx
        ]

        # Update gate tracking
        self._gates_passed[
            torch.arange(self.num_envs, device=self.device), next_idx
        ] |= just_passed
        self._current_gate_idx += just_passed.long()

        # Check course completion
        all_done = self._current_gate_idx >= self._active_num_gates

        # Contact forces for collision detection
        contact = self._drone.data.net_contact_forces  # (N, num_bodies, 3)
        contact_mag = torch.norm(contact, dim=-1)      # (N, num_bodies)
        collided = contact_mag.max(dim=-1).values > self.cfg.collision_force_threshold

        # Compute total reward
        reward = compute_total_reward(
            dist_to_gate=dist_to_gate,
            prev_dist_to_gate=self._prev_dist_to_gate,
            passed_gate=just_passed,
            all_gates_passed=all_done,
            collided=collided,
            current_action=self._raw_actions,
            previous_action=self._prev_action,
            progress_scale=self.cfg.progress_scale,
            gate_bonus=self.cfg.gate_pass_bonus,
            completion_bonus=self.cfg.course_completion_bonus,
            time_pen=self.cfg.time_penalty,
            smooth_scale=self.cfg.smoothness_scale,
            collision_pen=self.cfg.collision_penalty,
        )

        # Update state for next step
        self._prev_dist_to_gate = dist_to_gate
        self._prev_action = self._raw_actions.clone()
        self._step_count += 1

        return reward

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute terminated (failure) and truncated (timeout) flags."""
        pos = self._drone.data.root_state_w[:, :3]
        contact = self._drone.data.net_contact_forces
        contact_mag = torch.norm(contact, dim=-1)

        collided = contact_mag.max(dim=-1).values > self.cfg.collision_force_threshold
        oob = check_out_of_bounds(
            pos, self.cfg.geofence_radius, self.cfg.min_altitude, self.cfg.max_altitude
        )
        course_done = self._current_gate_idx >= self._active_num_gates

        terminated = collided | oob | course_done
        truncated = check_timeout(self._step_count, int(self.cfg.episode_length_s / self.cfg.sim_dt / self.cfg.decimation))
        truncated = truncated & ~terminated

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset selected environments with randomized drone pose and gate layout."""
        super()._reset_idx(env_ids)
        num_reset = len(env_ids)

        # Reset tracking state
        self._step_count[env_ids] = 0
        self._prev_action[env_ids] = 0.0
        self._current_gate_idx[env_ids] = 0
        self._gates_passed[env_ids] = False

        # Generate new track layout for each reset env
        gen_fn = _TRACK_GENERATORS.get(self.cfg.track_type, generate_zigzag)
        for i, env_id in enumerate(env_ids.tolist()):
            self._track_seed += 1
            track = gen_fn(
                num_gates=self.cfg.num_gates,
                jitter=self.cfg.gate_jitter,
                seed=self._track_seed,
            )
            for g_idx, gate_pose in enumerate(track.gates):
                if g_idx >= self.cfg.max_num_gates:
                    break
                gate_pos = torch.tensor(gate_pose.position, device=self.device)
                self._gate_positions[env_id, g_idx] = gate_pos

                # Move the gate rigid body in the scene
                if g_idx < len(self._gate_handles):
                    self._gate_handles[g_idx].write_root_state_to_sim(
                        root_state=torch.tensor(
                            [gate_pos[0], gate_pos[1], gate_pos[2],
                             1.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            device=self.device,
                        ).unsqueeze(0),
                        env_ids=torch.tensor([env_id], device=self.device),
                    )

            self._active_num_gates[env_id] = min(len(track.gates), self.cfg.max_num_gates)

        # Randomize drone initial state
        drone_pos = torch.zeros(num_reset, 3, device=self.device)
        drone_pos[:, 2] = 1.5  # hover altitude
        # Small random offset
        drone_pos[:, :2] += (torch.rand(num_reset, 2, device=self.device) - 0.5) * 2.0

        drone_quat = torch.zeros(num_reset, 4, device=self.device)
        drone_quat[:, 0] = 1.0  # identity quaternion

        drone_vel = torch.zeros(num_reset, 6, device=self.device)

        root_state = torch.cat([drone_pos, drone_quat, drone_vel], dim=-1)
        self._drone.write_root_state_to_sim(root_state, env_ids)

        # Initialize distance tracking
        next_gate_pos = self._gate_positions[env_ids, 0]
        self._prev_dist_to_gate[env_ids] = torch.norm(
            drone_pos - next_gate_pos, dim=-1
        )
