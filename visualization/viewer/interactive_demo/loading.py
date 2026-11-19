# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from motion_mamba2.model.assets import skeleton_asset_path

from .common import *  # noqa: F401,F403


class ModelLoadingMixin:
    _remote_load_lock = threading.Lock()
    def _build_text_encoder(self):
        """Build the shared text encoder once, at demo startup.

        Uses "auto" mode (connect to the remote service if reachable, otherwise load the encoder in-
        process) with the service URL taken from the TEXT_ENCODER_URL env var. Built once and reused
        across every model load (core / g1 / soma) and across clients, so the ~16GB local encoder is
        never loaded more than once; the result is passed into ``load_model(..., text_encoder=...)``
        on each load.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return load_text_encoder(mode="api", device=device)

    def get_skeleton_info(self, model_skeleton):
        """Detect skeleton type from model and return appropriate class and mesh mode.

        Args:
            model_skeleton: The skeleton from model.motion_rep.skeleton

        Returns:
            tuple: (skeleton_class, mesh_mode, skeleton_name)
        """
        # Check by type first
        if isinstance(model_skeleton, G1Skeleton34):
            return G1Skeleton34, "g1_stl", "g1skel34"
        elif isinstance(model_skeleton, CoreSkeleton27):
            return CoreSkeleton27, "core_skin", "cskel27"
        elif isinstance(model_skeleton, (SOMASkeleton30, SOMASkeleton77)):
            skel_class = SOMASkeleton30 if model_skeleton.nbjoints == 30 else SOMASkeleton77
            return skel_class, "soma_skin", model_skeleton.name

        # Fallback to name-based detection
        skeleton_name = getattr(model_skeleton, "name", "")
        if skeleton_name.startswith("g1skel"):
            return G1Skeleton34, "g1_stl", skeleton_name
        elif skeleton_name.startswith("cskel"):
            return CoreSkeleton27, "core_skin", skeleton_name
        elif skeleton_name.startswith("somaskel"):
            skel_class = SOMASkeleton30 if "30" in skeleton_name else SOMASkeleton77
            return skel_class, "soma_skin", skeleton_name

        # Default to CoreSkeleton27
        print(f"Warning: Unknown skeleton type, defaulting to CoreSkeleton27")
        return CoreSkeleton27, "core_skin", "cskel27"

    def load_model(self, client_id: int, model_name: str, progress=None, *, take_control=False):
        if os.environ.get("ARDY_GENERATOR_BACKEND") == "motion_mamba":
            with self._remote_load_lock:
                if take_control and self.client_active(client_id):
                    from .remote_control_owner import release_remote_owners
                    release_remote_owners(self, client_id)
                return self._load_model_impl(client_id, model_name, progress)
        return self._load_model_impl(client_id, model_name, progress)

    def _load_model_impl(self, client_id: int, model_name: str, progress=None):
        """Load the motion generation model for a specific client.

        ``model_name`` is a released model name (dropdown value). It is loaded from CHECKPOINTS_DIR
        when set, otherwise downloaded from Hugging Face.

        ``progress`` is an optional ``callable(message: str)`` used to surface each loading phase
        (download / TRT export / engine build) to the caller — the demo passes one that updates the
        on-screen loading notification.
        """
        if not self.client_active(client_id):
            return None

        session = self.client_sessions[client_id]
        remote = os.environ.get("ARDY_GENERATOR_BACKEND") == "motion_mamba"

        def _progress(message: str) -> None:
            """Log a loading-phase message to the console and, when a progress callback was
            supplied, forward it to the GUI (loading notification)."""
            print(message, flush=True)
            if progress is not None:
                progress(message)

        print(f"Loading model '{model_name}' for client {client_id}...")

        # Reuse the shared text encoder built once at startup. Passing
        # checkpoints_dir=None makes load_model download from Hugging Face.
        if remote:
            _progress("Connecting to the isolated Motion Mamba 2.0 generator...")
        elif CHECKPOINTS_DIR:
            _progress(f"Loading model '{model_name}' from checkpoints...")
        else:
            _progress(f"Downloading model '{model_name}' from Hugging Face...")
        try:
            if remote:
                from .motion_mamba_backend import RemoteMotionMambaModel
                if any(getattr(s.model, "is_remote_mm2", False) for cid, s in self.client_sessions.items() if cid != client_id):
                    session.mm2_control_blocked = True
                    session.gui_elements.gui_load_model_button.label = "Take Control"
                    session.gui_elements.gui_chosen_model_md.content = (
                        "**Model:** " + "MotionMamba2"
                        + " — controlled by another tab. Click **Take Control** to use this tab."
                    )
                    _progress("Another tab controls SONIC. Click Take Control in the Model tab.")
                    return None
                session.mm2_control_blocked = False
                if getattr(session.model, "is_remote_mm2", False):
                    session.model.close()
                model = RemoteMotionMambaModel(self.text_encoder)
                model_cfg = None
                _progress(f"Connected to {model.model_label} (G1, 25 FPS, 52 frames)")
            else:
                model, model_cfg = load_model(
                    model_name,
                    device=self.device,
                    return_config=True,
                    text_encoder=self.text_encoder,
                    checkpoints_dir=CHECKPOINTS_DIR,
                )
        except Exception as e:
            print(f"Failed to load model '{model_name}': {e}")
            if remote:
                session.client.add_notification(title="Motion Mamba backend", body=str(e), color="red", auto_close_seconds=15)
            return None

        # Local folder for engine caching (local dir or cached HF snapshot).
        model_dir = None if remote or os.environ.get("MM2_CHECKPOINT") else resolve_model_dir(model_name)

        model_fps = model.motion_rep.fps
        print(f"Model FPS: {model_fps}")

        # Update session with new model and attributes
        session.model = model
        session.motion_rep = model.motion_rep
        # create motion_rep_infer from motion_rep
        # Detect skeleton type from loaded model
        skeleton_class, mesh_mode, skeleton_name = self.get_skeleton_info(model.motion_rep.skeleton)
        print(f"Detected skeleton: {skeleton_name} (class: {skeleton_class.__name__}, mesh_mode: {mesh_mode})")
        prev_skeleton_name = session.gui_elements.gui_skeleton_name_text.value
        session.gui_elements.gui_skeleton_name_text.value = skeleton_name

        # The Core skeleton has no companion bones-seed dataset for constraint
        # sampling; gate the relevant buttons accordingly. The click-time guard
        # in gui/generate.py (and its "z"-shortcut duplicate) stays as a backstop.
        if skeleton_supports_constraint_sampling(model.motion_rep.skeleton):
            session.gui_elements.gui_load_seq_button.disabled = False
            session.gui_elements.gui_load_seq_button.label = "Sample Constraints"
            session.gui_elements.gui_random_motion_button.disabled = False
        else:
            session.gui_elements.gui_load_seq_button.disabled = True
            session.gui_elements.gui_load_seq_button.label = "Sample Constraints (unavailable for Core)"
            session.gui_elements.gui_random_motion_button.disabled = True

        skeleton_infer = model.skeleton
        session.motion_rep = model.motion_rep

        # Create MujocoQposConverter for Mujoco/CSV motion import (G1 skeleton only)
        is_g1 = skeleton_class is G1Skeleton34 or is_g1_skeleton(skeleton_infer)
        print(f"[Mujoco] skeleton_class={skeleton_class.__name__}, nbjoints={skeleton_infer.nbjoints}, is_g1={is_g1}")
        if is_g1:
            try:
                from motion_mamba2.model.mujoco_export import MujocoQposConverter

                # Find g1.xml: try skeleton folder first, then ardy package assets
                xml_path = os.path.join(skeleton_infer.folder, "xml", "g1.xml")
                if not os.path.exists(xml_path):
                    xml_path = str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
                print(f"[Mujoco] MujocoQposConverter xml_path={xml_path}, exists={os.path.exists(xml_path)}")
                session.mujoco_converter = MujocoQposConverter(skeleton_infer, xml_path=xml_path)
                print("[Mujoco] MujocoQposConverter initialized")
            except Exception as e:
                print(f"[Mujoco] Could not create MujocoQposConverter: {e}")
                import traceback

                traceback.print_exc()
                session.mujoco_converter = None
        else:
            print(f"[Mujoco] Skipping MujocoQposConverter — not a G1 skeleton")
            session.mujoco_converter = None

        # Clear reference motion from previous model
        if session.ref_character is not None:
            session.ref_character.clear()
            session.ref_character = None
        session.ref_joints_pos = None
        session.ref_joints_rot = None

        # Update default motion file path based on skeleton type
        if is_g1:
            session.gui_elements.gui_motion_file_path.value = (
                "datasets/bones-seed/g1/csv/230306/jog_ff_loop_180_R_001__A244.csv"
            )
        else:
            session.gui_elements.gui_motion_file_path.value = (
                "datasets/bones-seed/soma_uniform/bvh/230306/jog_ff_loop_180_R_001__A244.bvh"
            )

        # Store mesh_mode for character creation
        session.mesh_mode = mesh_mode
        session.model_fps = model_fps
        session.num_frames_per_token = model.denoiser.num_frames_per_token
        session.gen_horizon_len = model.gen_horizon_len

        # Update GUI elements
        session.gui_elements.gui_model_fps.value = model_fps
        session.gui_elements.gui_diffusion_steps_slider.value = model.diffusion.num_base_steps
        patch = session.num_frames_per_token
        # Total per-step window budget: 10 s of frames, rounded down to a token
        # multiple. TRT engines are built exactly this size (see max_tok below),
        # so no generation step may request a larger window.
        session.max_window_len = (10 * model_fps // patch) * patch
        # Round down to a multiple of the step size (patch) so the slider's
        # discrete steps land exactly on the max.
        crop_max = ((session.max_window_len - session.gen_horizon_len) // patch) * patch

        # History Crop Length: min=patch, max=crop_max, step=patch, default=min
        session.gui_elements.gui_history_crop_length.min = patch
        session.gui_elements.gui_history_crop_length.max = crop_max
        session.gui_elements.gui_history_crop_length.step = patch
        session.gui_elements.gui_history_crop_length.value = patch

        # Future Crop Length: min=0, max=crop_max, step=patch, default=max
        session.gui_elements.gui_future_crop_length.min = 0
        session.gui_elements.gui_future_crop_length.max = crop_max
        session.gui_elements.gui_future_crop_length.step = patch
        session.gui_elements.gui_future_crop_length.value = crop_max

        session.gui_elements.gui_waypoint_interval.value = 3 * model_fps

        # Update timeline fps
        if hasattr(session.client, "timeline"):
            session.client.timeline._fps = model_fps
            print(f"Timeline FPS set to {model_fps}")

        # Constraints are tied to a specific rig (joint/bone layout). If the
        # skeleton type changed (e.g. G1 -> SOMA), the existing constraints can't
        # be reinterpreted on the new rig, so clear them. Within the same skeleton
        # (e.g. a different horizon) keep them and just repoint at the new skeleton.
        if prev_skeleton_name and prev_skeleton_name != skeleton_name:
            self.clear_constraints(client_id)
        # Update constraint tracks with the model's skeleton
        for constraint in session.constraints.values():
            constraint.skeleton = session.motion_rep.skeleton

        if remote:
            model.configure(self, session)
        print(f"Model loaded successfully for client {client_id}!")
        return model
