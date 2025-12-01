#!/usr/bin/env python3
"""
orbit_sandbox.py

Fictional-system orbit sandbox.

This version has:

- Local SYSTEM view (what you already had):
    * Top-down / isometric (V).
    * Orbit-relative time scaling (inner faster, outer slower).
    * Left-drag to pan, tap to select body.
    * Moon mini-panel for the focused giant (M), with clickable moons.
    * Wiki-ish sidebar with meta for the focused body.
    * Distance measuring tool (D + drag).
    * Optional camera follow (F) for the focused body.

- CLUSTER view:
    * Stars = systems (RR, Tengri, Gargan, Octaeva), placed in a local “bubble”.
    * Positions come from `offset` (or `pos_ly`) in universe.json.
    * Left-drag to pan, zoom as usual.
    * Click a star to “dive into” that system (back to SYSTEM view).
    * D + drag still measures distances in LY, with light-time and travel times.

Toggles
-------
G: toggle between SYSTEM view (local orbits) and CLUSTER view (star map)
, / . : previous / next system in the universe.json list

Usage
-----
    python orbit_sandbox.py data/universe.json --system rr

Dependencies
-----------
    pip install pygame
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import pygame

AU_IN_KM = 149_597_870.7
LY_IN_KM = AU_IN_KM * 63241.077  # ~1 light-year in km
SHIP_SPEED_KM_S = 20.0
SPEED_OF_LIGHT_KM_S = 299_792.458
BASE_ORBIT_SECONDS = 20.0  # real seconds for one full orbit at speed x1


# ---------- Data model ----------


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert '#rrggbb' or 'rrggbb' to an (r, g, b) tuple."""
    if not hex_str:
        return (255, 255, 255)
    s = hex_str.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return (255, 255, 255)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


class UIButton:
    """Simple rectangular UI button with hover and click handling."""
    def __init__(self, label: str, x: int, y: int, w: int, h: int, font: pygame.font.Font, callback):
        self.label = label
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.callback = callback
        self.hover = False

    def draw(self, surface: pygame.Surface):
        bg = (50, 54, 70) if not self.hover else (80, 90, 130)
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, (120, 130, 170), self.rect, 2, border_radius=6)
        txt = self.font.render(self.label, True, (230, 230, 250))
        tx = self.rect.x + (self.rect.w - txt.get_width()) // 2
        ty = self.rect.y + (self.rect.h - txt.get_height()) // 2
        surface.blit(txt, (tx, ty))

    def contains(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)

    def click(self):
        if callable(self.callback):
            self.callback()


class Body:
    def __init__(self, raw: dict, system_id: str):
        self.id: str = raw["id"]
        self.name: str = raw.get("name", self.id)
        self.type: str = raw.get("type", "planet")
        self.system: str = raw.get("system", system_id)
        self.parent_id: Optional[str] = raw.get("parent")
        self.parent: Optional["Body"] = None
        self.children: List["Body"] = []

        # Orbital parameters
        self.a_au: float = float(raw.get("a", 0.0))
        self.e: float = float(raw.get("e", 0.0))
        self.inclination_deg: float = float(raw.get("inclination", 0.0))

        # Physical / visual
        self.radius_km: float = float(raw.get("radius", 0.0))
        self.visual_size: int = int(raw.get("visual_size", 4))
        self.color: Tuple[int, int, int] = hex_to_rgb(raw.get("color", "#ffffff"))

        # Orbital period
        period = raw.get("period_years")
        if period is None and self.a_au > 0:
            self.period_years: float = self.a_au ** 1.5
        else:
            self.period_years = float(period or 0.0)

        # Initial phase
        phase_deg = raw.get("phase_deg")
        if phase_deg is None:
            self.initial_phase: float = random.random() * 2.0 * math.pi
        else:
            self.initial_phase = math.radians(float(phase_deg))

        self.angle: float = self.initial_phase
        self.mean_motion: float = (
            2.0 * math.pi / self.period_years if self.period_years > 0 else 0.0
        )

        # Lore / meta
        self.tags: List[str] = raw.get("tags", [])
        self.image: Optional[str] = raw.get("image")
        self.meta: dict = raw.get("meta", {})

        # Position in world coords (AU for system mode)
        self.pos: Tuple[float, float] = (0.0, 0.0)

    def is_root(self) -> bool:
        return self.parent is None

    def is_belt(self) -> bool:
        return self.type in ("belt", "asteroid_belt")

    def is_moon(self) -> bool:
        return self.type in ("moon", "satellite")

    def update_angle(self, dt_years: float):
        if self.is_root() or self.mean_motion == 0.0:
            return
        self.angle = (self.angle + self.mean_motion * dt_years) % (2.0 * math.pi)


class SystemModel:
    def __init__(self, system_id: str, name: str, bodies_data: List[dict]):
        self.id = system_id
        self.name = name
        self.bodies: Dict[str, Body] = {}

        for raw in bodies_data:
            b = Body(raw, system_id)
            self.bodies[b.id] = b

        # Link parents/children
        for b in self.bodies.values():
            if b.parent_id:
                parent = self.bodies.get(b.parent_id)
                if parent:
                    b.parent = parent
                    parent.children.append(b)

        self.roots: List[Body] = [b for b in self.bodies.values() if b.is_root()]
        if not self.roots:
            raise ValueError("System must have at least one root (e.g., barycenter/star).")

        max_a = 0.0
        for b in self.bodies.values():
            if not b.is_root():
                max_a = max(max_a, b.a_au)
        self.max_a_au = max_a if max_a > 0 else 1.0

    def update(self, dt_years: float):
        for b in self.bodies.values():
            b.update_angle(dt_years)
        for root in self.roots:
            self._update_positions_recursive(root)

    def _update_positions_recursive(self, body: Body):
        if body.is_root():
            body.pos = (0.0, 0.0)
        else:
            px, py = body.parent.pos
            r = body.a_au
            body.pos = (px + r * math.cos(body.angle), py + r * math.sin(body.angle))
        for child in body.children:
            self._update_positions_recursive(child)

    def ordered_focusable_bodies(self) -> List[Body]:
        focusables = [
            b for b in self.bodies.values()
            if not b.is_belt() and b.type != "barycenter"
        ]
        focusables.sort(key=lambda b: (b.type not in ("star", "primary_star"), b.a_au))
        return focusables


# ---------- JSON loading ----------


def load_system_from_json(path: str, system_id: Optional[str] = None) -> SystemModel:
    """Load a single system from either system.json or universe.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "systems" in data and isinstance(data["systems"], list):
        systems = data["systems"]
        if not systems:
            raise ValueError("Universe JSON has no systems.")

        chosen = None
        if system_id:
            for s in systems:
                if s.get("id") == system_id:
                    chosen = s
                    break
            if chosen is None:
                raise ValueError(f"System id '{system_id}' not found in universe.")
        else:
            chosen = systems[0]

        sid = chosen.get("id", "system")
        name = chosen.get("name", sid)

        if "bodies" in chosen:
            bodies_data = chosen["bodies"]
        else:
            bodies_file = chosen.get("bodies_file") or chosen.get("file")
            if not bodies_file:
                raise ValueError("Universe system missing 'bodies' or 'bodies_file'.")
            base_dir = os.path.dirname(path)
            bodies_path = os.path.join(base_dir, bodies_file)
            with open(bodies_path, "r", encoding="utf-8") as bf:
                bodies_json = json.load(bf)
                bodies_data = bodies_json["bodies"]

        return SystemModel(sid, name, bodies_data)

    # Single system file
    sid = data.get("id", "system")
    name = data.get("name", sid)
    bodies_data = data["bodies"]
    return SystemModel(sid, name, bodies_data)


def list_systems_in_universe(path: str) -> List[dict]:
    """
    Return a list of system descriptors from universe.json:
      [{ "id": ..., "name": ..., "offset": (x,y), "color": (r,g,b) }, ...]
    offset is in 'map units' (we'll treat them as light-years for the cluster view).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []

    systems = []
    raw_systems = data.get("systems")
    if not isinstance(raw_systems, list):
        return []

    for s in raw_systems:
        sid = s.get("id")
        if not sid:
            continue
        name = s.get("name", sid)
        offset = s.get("offset") or s.get("pos_ly") or [0.0, 0.0]
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            offset = [0.0, 0.0]
        try:
            ox = float(offset[0])
            oy = float(offset[1])
        except Exception:
            ox, oy = 0.0, 0.0
        color_hex = s.get("color", "#ffffff")
        color = hex_to_rgb(color_hex)
        systems.append(
            {
                "id": sid,
                "name": name,
                "offset": (ox, oy),
                "color": color,
            }
        )
    return systems


# ---------- Viewer ----------


class OrbitSandbox:
    def __init__(
        self,
        system: SystemModel,
        json_path: str,
        system_defs: Optional[List[dict]] = None,
        current_system_id: Optional[str] = None,
        width: int = 1200,
        height: int = 700,
    ):
        pygame.init()
        pygame.display.set_caption("orbit_sandbox")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Universe context
        self.json_path = json_path
        self.system_defs = system_defs or []
        self.current_system_id = current_system_id or getattr(system, "id", "system")

        # Mode: "system" (local orbits) vs "cluster" (star map)
        # Start in cluster-first mode so the user lands in the local cluster view.
        self.map_mode = "cluster"
        self.view_mode = "top"      # "top" or "iso"
        self.scope_mode = "system"  # "system" or "local" (moon panel)

        # Simulation
        self.system = system
        self.sim_time_years = 0.0
        self.running = True

        # Time scaling (orbit-relative)
        self.base_time_scale = 1.0
        self.speed_multiplier = 1.0
        self.time_scale = 1.0

        # Camera
        self.cam_x = 0.0
        self.cam_y = 0.0

        # Scaling
        self.base_pixels_per_au = (
            0.44 * min(self.width, self.height) / self.system.max_a_au
        )

        if self.system_defs:
            xs = [sd["offset"][0] for sd in self.system_defs]
            ys = [sd["offset"][1] for sd in self.system_defs]
            max_extent = max(
                max(abs(x) for x in xs),
                max(abs(y) for y in ys),
                1.0,
            )
        else:
            max_extent = 1.0
        self.cluster_pixels_per_unit = 0.4 * min(self.width, self.height) / max_extent

        self.zoom = 1.0

        # Fonts
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 20, bold=True)

        # Toggles
        self.show_belts = True
        self.show_moons = True
        self.show_dwarfs = True
        self.follow_focus = False
        # Show wiki/info sidebar in cluster view by default
        self.show_wiki_sidebar = True
        # Time scale modes: 'focus' ties the time scale to the focused body's period
        # (legacy behaviour), 'global' uses a stable global years-per-second so
        # very large orbital periods do not increase apparent simulation speed.
        self.time_scale_mode = "global"
        # When in 'global' mode, this is the simulated years advanced per real second
        self.global_time_scale = 0.05

        # Focusables
        self.focusables = self.system.ordered_focusable_bodies()
        self.focus_index = 0 if self.focusables else -1
        self.focus_body: Optional[Body] = (
            self.focusables[self.focus_index] if self.focus_index >= 0 else None
        )

        if self.focus_body:
            self.cam_x, self.cam_y = self.focus_body.pos

        # Drag / pan
        self.left_dragging = False
        self.left_drag_start_screen: Tuple[int, int] = (0, 0)
        self.left_drag_start_cam: Tuple[float, float] = (self.cam_x, self.cam_y)

        # Measurement
        self.measure_mode = False
        self.measure_dragging = False
        self.measure_start_world: Optional[Tuple[float, float]] = None
        self.measure_end_world: Optional[Tuple[float, float]] = None

        # Moon panel click state
        self.moon_panel_state: Optional[dict] = None

        # UI buttons for cluster view
        self.ui_buttons: List[UIButton] = []
        self.hovered_system_idx: Optional[int] = None
        self._init_ui_buttons()

        self.recalc_time_scale()

    # -------- time & scale helpers --------

    def _init_ui_buttons(self):
        """Initialize UI buttons based on current view mode."""
        sidebar_x = int(self.width * 0.68)
        bx = sidebar_x + 20
        by = self.height - 280
        bw = int((self.width - sidebar_x) * 0.85)
        bh = 40
        f = self.font

        if self.map_mode == "cluster":
            self.ui_buttons = [
                UIButton("Explore System", bx, by, bw, bh, f, self._btn_explore),
                UIButton("Toggle Wiki", bx, by + 50, bw, bh, f, self._btn_wiki),
                UIButton("Measure", bx, by + 100, bw, bh, f, self._btn_measure),
                UIButton("Reset View", bx, by + 150, bw, bh, f, self._btn_reset),
            ]
        elif self.map_mode == "body":
            self.ui_buttons = [
                UIButton("Return to System", bx, by, bw, bh, f, self._btn_return_system),
                UIButton("View Wiki", bx, by + 50, bw, bh, f, self._btn_wiki),
                UIButton("Reset View", bx, by + 100, bw, bh, f, self._btn_reset),
            ]
        elif self.map_mode == "wiki":
            self.ui_buttons = [
                UIButton("Return to System", bx, by, bw, bh, f, self._btn_return_system),
                UIButton("Reset View", bx, by + 50, bw, bh, f, self._btn_reset),
            ]
        else:  # system view
            self.ui_buttons = [
                UIButton("Return to Cluster", bx, by, bw, bh, f, self._btn_return_cluster),
                UIButton("Body Detail", bx, by + 50, bw, bh, f, self._btn_body_detail),
                UIButton("Toggle Wiki", bx, by + 100, bw, bh, f, self._btn_wiki),
                UIButton("Measure", bx, by + 150, bw, bh, f, self._btn_measure),
            ]

    def _btn_return_cluster(self):
        """Return to cluster view."""
        self.map_mode = "cluster"
        # Reset camera to center on the cluster
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.zoom = 1.0
        # Reset any measurement mode
        self.measure_mode = False
        self.measure_dragging = False
        self.measure_start_world = None
        self.measure_end_world = None
        self._init_ui_buttons()

    def _btn_body_detail(self):
        """Show detailed body view."""
        if self.focus_body:
            self.map_mode = "body"
            self._init_ui_buttons()

    def _btn_return_system(self):
        """Return to system view from body view."""
        self.map_mode = "system"
        self._init_ui_buttons()

    def _btn_explore(self):
        """Dive into the hovered or current system."""
        # Use hovered system if available, otherwise use current system
        idx = self.hovered_system_idx
        if idx is None:
            # Find current system index
            for i, sd in enumerate(self.system_defs):
                if sd["id"] == self.current_system_id:
                    idx = i
                    break
        
        if idx is not None and 0 <= idx < len(self.system_defs):
            sd = self.system_defs[idx]
            self.load_system_by_id(sd["id"])
            self.map_mode = "system"
            self._init_ui_buttons()  # Refresh buttons for system view

    def _btn_wiki(self):
        """Navigate to wiki view."""
        if self.map_mode == "wiki":
            # Return to previous mode (system or cluster)
            # Default to system if we have a loaded system
            self.map_mode = "system" if self.system else "cluster"
        else:
            # Go to wiki view
            self.map_mode = "wiki"
        self._init_ui_buttons()

    def _btn_measure(self):
        """Toggle measurement mode."""
        self.measure_mode = not self.measure_mode
        if not self.measure_mode:
            self.measure_dragging = False
            self.measure_start_world = None
            self.measure_end_world = None

    def _btn_reset(self):
        """Reset camera and zoom."""
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.zoom = 1.0

    def current_scale(self) -> float:
        if self.map_mode == "system":
            return self.base_pixels_per_au * self.zoom
        else:
            return self.cluster_pixels_per_unit * self.zoom

    def recalc_time_scale(self):
        """Adjust time_scale according to selected mode.

        Modes:
          - 'focus': keep legacy behaviour (focused body's full orbit ~= BASE_ORBIT_SECONDS)
          - 'global': use fixed `global_time_scale` so large orbital periods don't increase overall speed
        """
        b = self.focus_body
        if self.time_scale_mode == "focus":
            if b and b.period_years > 0:
                self.base_time_scale = b.period_years / BASE_ORBIT_SECONDS
            else:
                self.base_time_scale = 1.0
        else:
            # Stable global scale: years simulated per real second
            # Use the configured global_time_scale, but ensure it's positive
            try:
                self.base_time_scale = float(self.global_time_scale)
            except Exception:
                self.base_time_scale = 0.05
        self.time_scale = self.base_time_scale * self.speed_multiplier

    # -------- coordinate transforms --------

    def world_to_screen(self, wx: float, wy: float, scale: float) -> Tuple[float, float]:
        dx = wx - self.cam_x
        dy = wy - self.cam_y
        if self.view_mode == "top":
            sx = self.width / 2 + dx * scale
            sy = self.height / 2 + dy * scale
        else:
            iso_x = (dx - dy) * scale * 0.75
            iso_y = (dx + dy) * scale * 0.40
            sx = self.width / 2 + iso_x
            sy = self.height / 2 + iso_y
        return sx, sy

    def screen_to_world(self, sx: float, sy: float, scale: float) -> Tuple[float, float]:
        if self.view_mode == "top":
            dx = (sx - self.width / 2) / scale
            dy = (sy - self.height / 2) / scale
        else:
            X = (sx - self.width / 2) / scale
            Y = (sy - self.height / 2) / scale
            dx_minus_dy = X / 0.75
            dx_plus_dy = Y / 0.40
            dx = (dx_minus_dy + dx_plus_dy) / 2.0
            dy = (dx_plus_dy - dx_minus_dy) / 2.0
        wx = self.cam_x + dx
        wy = self.cam_y + dy
        return wx, wy

    # -------- main loop --------

    def run(self):
        while True:
            dt_real = self.clock.tick(60) / 1000.0
            self.handle_events()
            if self.running:
                if self.map_mode == "system":
                    dt_years = dt_real * self.time_scale
                    self.sim_time_years += dt_years
                    self.system.update(dt_years)
                    if self.follow_focus and self.focus_body and not self.left_dragging:
                        # Smooth follow
                        tx, ty = self.focus_body.pos
                        alpha = 0.12
                        self.cam_x += (tx - self.cam_x) * alpha
                        self.cam_y += (ty - self.cam_y) * alpha
            self.draw()

    # -------- system switching --------

    def load_system_by_id(self, system_id: str):
        new_system = load_system_from_json(self.json_path, system_id=system_id)
        new_system.update(0.0)
        self.system = new_system
        self.current_system_id = system_id

        self.base_pixels_per_au = (
            0.44 * min(self.width, self.height) / self.system.max_a_au
        )

        self.focusables = self.system.ordered_focusable_bodies()
        self.focus_index = 0 if self.focusables else -1
        self.focus_body = self.focusables[0] if self.focus_index >= 0 else None
        if self.focus_body:
            self.cam_x, self.cam_y = self.focus_body.pos
        else:
            self.cam_x, self.cam_y = 0.0, 0.0

        self.sim_time_years = 0.0
        self.measure_mode = False
        self.measure_dragging = False
        self.measure_start_world = None
        self.measure_end_world = None
        self.recalc_time_scale()

    def switch_system_relative(self, delta: int):
        if not self.system_defs:
            return
        cur_id = self.current_system_id
        ids = [sd["id"] for sd in self.system_defs]
        try:
            idx = ids.index(cur_id)
        except ValueError:
            idx = 0
        idx = (idx + delta) % len(ids)
        new_id = ids[idx]
        self.load_system_by_id(new_id)

    # -------- event handling --------

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_left_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.handle_left_up(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.zoom *= 1.1
                elif event.y < 0:
                    self.zoom /= 1.1
                self.zoom = max(0.1, min(self.zoom, 10.0))

    def handle_keydown(self, key):
        # basic sim controls
        if key == pygame.K_SPACE:
            self.running = not self.running

        if key == pygame.K_UP:
            self.cycle_focus(1)
        if key == pygame.K_DOWN:
            self.cycle_focus(-1)
        if key == pygame.K_TAB:
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.cycle_focus(-1)
            else:
                self.cycle_focus(1)

        # system cycling (universe.json order)
        if key == pygame.K_COMMA:
            self.switch_system_relative(-1)
        if key == pygame.K_PERIOD:
            self.switch_system_relative(1)

        # zoom
        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.zoom *= 1.1
        if key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.zoom /= 1.1
        self.zoom = max(0.1, min(self.zoom, 10.0))

        # time scale
        if key == pygame.K_RIGHTBRACKET:
            self.speed_multiplier *= 2.0
            self.speed_multiplier = min(self.speed_multiplier, 128.0)
            self.recalc_time_scale()
        if key == pygame.K_LEFTBRACKET:
            self.speed_multiplier /= 2.0
            self.speed_multiplier = max(self.speed_multiplier, 1.0 / 128.0)
            self.recalc_time_scale()

        if key == pygame.K_0:
            self.zoom = 1.0
            if self.map_mode == "system" and self.focus_body:
                self.cam_x, self.cam_y = self.focus_body.pos
            else:
                self.cam_x, self.cam_y = 0.0, 0.0

        # toggles
        if key == pygame.K_1:
            self.show_belts = not self.show_belts
        if key == pygame.K_2:
            self.show_moons = not self.show_moons
        if key == pygame.K_3:
            self.show_dwarfs = not self.show_dwarfs

        if key == pygame.K_v:
            self.view_mode = "iso" if self.view_mode == "top" else "top"

        if key == pygame.K_m:
            # Navigate to body detail view if in system mode with a focused body
            if self.map_mode == "system" and self.focus_body:
                self.map_mode = "body"
                self._init_ui_buttons()
            elif self.map_mode == "body":
                # Return to system view from body view
                self.map_mode = "system"
                self._init_ui_buttons()

        # measure mode
        if key == pygame.K_d:
            self.measure_mode = not self.measure_mode
            if not self.measure_mode:
                self.measure_dragging = False
                self.measure_start_world = None
                self.measure_end_world = None

        # follow focus
        if key == pygame.K_f:
            self.follow_focus = not self.follow_focus

        # map mode (system vs cluster)
        if key == pygame.K_g:
            self.map_mode = "cluster" if self.map_mode == "system" else "system"

        # toggle wiki/info sidebar in cluster view
        if key == pygame.K_w:
            self.show_wiki_sidebar = not self.show_wiki_sidebar

        # toggle time-scale mode between 'global' and 'focus'
        if key == pygame.K_t:
            self.time_scale_mode = "focus" if self.time_scale_mode == "global" else "global"
            self.recalc_time_scale()
    # --- mouse controls ---

    def handle_left_down(self, pos):
        mx, my = pos
        scale = self.current_scale()
        if self.measure_mode:
            self.measure_dragging = True
            self.measure_start_world = self.screen_to_world(mx, my, scale)
            self.measure_end_world = self.measure_start_world
        else:
            self.left_dragging = True
            self.left_drag_start_screen = (mx, my)
            self.left_drag_start_cam = (self.cam_x, self.cam_y)

    def handle_left_up(self, pos):
        mx, my = pos
        scale = self.current_scale()

        if self.measure_mode:
            if self.measure_dragging:
                self.measure_dragging = False
                self.measure_end_world = self.screen_to_world(mx, my, scale)
        else:
            if self.left_dragging:
                dx = mx - self.left_drag_start_screen[0]
                dy = my - self.left_drag_start_screen[1]
                d2 = dx * dx + dy * dy
                threshold2 = 16  # <= 4px = click, otherwise pan
                if d2 <= threshold2:
                    # Check UI buttons first (works for both cluster and system view)
                    for btn in self.ui_buttons:
                        if btn.contains(pos):
                            btn.click()
                            self.left_dragging = False
                            return
                    
                    if self.map_mode == "cluster":
                        # click on star in cluster map
                        sys_id = self.pick_system_at(mx, my, scale)
                        if sys_id:
                            self.load_system_by_id(sys_id)
                            self.map_mode = "system"
                            self._init_ui_buttons()  # Refresh buttons for system view
                    else:
                        # system mode: try moon panel first, then bodies
                        if not self.try_moon_panel_click(mx, my):
                            body = self.pick_body_at(mx, my, scale)
                            if body:
                                self.set_focus(body)
                self.left_dragging = False

    def handle_mouse_motion(self, pos):
        mx, my = pos
        scale = self.current_scale()

        if self.measure_mode and self.measure_dragging and self.measure_start_world:
            self.measure_end_world = self.screen_to_world(mx, my, scale)
            return

        if self.left_dragging:
            dx = mx - self.left_drag_start_screen[0]
            dy = my - self.left_drag_start_screen[1]
            self.cam_x = self.left_drag_start_cam[0] - dx / scale
            self.cam_y = self.left_drag_start_cam[1] - dy / scale

    # -------- picking / focus --------

    def pick_system_at(self, mx: int, my: int, scale: float) -> Optional[str]:
        """Pick a system node in cluster view."""
        if not self.system_defs:
            return None
        best_id = None
        best_d2 = float("inf")
        for sd in self.system_defs:
            ox, oy = sd["offset"]
            sx, sy = self.world_to_screen(ox, oy, scale)
            dx = sx - mx
            dy = sy - my
            d2 = dx * dx + dy * dy
            r = 20  # Larger hit radius for easier clicking
            if d2 <= r * r and d2 < best_d2:
                best_d2 = d2
                best_id = sd["id"]
        return best_id

    def pick_body_at(self, mx: int, my: int, scale: float) -> Optional[Body]:
        best_body = None
        best_score = float("inf")
        for b in self.system.bodies.values():
            if b.is_belt():
                continue
            # body hit
            sx, sy = self.world_to_screen(b.pos[0], b.pos[1], scale)
            dx = sx - mx
            dy = sy - my
            d2 = dx * dx + dy * dy
            body_r = max(14, b.visual_size + 10)
            score = float("inf")
            if d2 <= body_r * body_r:
                score = d2
            # orbit hit
            if b.parent and b.a_au > 0:
                px, py = self.world_to_screen(b.parent.pos[0], b.parent.pos[1], scale)
                d_center = math.hypot(mx - px, my - py)
                r_orbit = b.a_au * scale
                if r_orbit > 4:
                    diff = abs(d_center - r_orbit)
                    if diff < 10:
                        score = min(score, diff * diff)
            if score < best_score:
                best_score = score
                best_body = b
        if best_score == float("inf"):
            return None
        return best_body

    def try_moon_panel_click(self, mx: int, my: int) -> bool:
        state = self.moon_panel_state
        if not state:
            return False
        rect = state["rect"]
        if not rect.collidepoint(mx, my):
            return False
        # Check moons
        for moon, sx, sy in state["moons"]:
            dx = mx - sx
            dy = my - sy
            if dx * dx + dy * dy <= 10 * 10:
                self.set_focus(moon)
                return True
        # Click near center selects parent
        cx, cy = state["center"]
        parent = state["center_body"]
        if (mx - cx) ** 2 + (my - cy) ** 2 <= 12 * 12:
            self.set_focus(parent)
            return True
        return False

    def cycle_focus(self, direction: int):
        if not self.focusables:
            return
        self.focus_index = (self.focus_index + direction) % len(self.focusables)
        self.set_focus(self.focusables[self.focus_index])

    def set_focus(self, body: Body):
        self.focus_body = body
        if body in self.focusables:
            self.focus_index = self.focusables.index(body)
        self.cam_x, self.cam_y = body.pos
        self.recalc_time_scale()

    # -------- drawing --------

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.moon_panel_state = None  # reset this frame

        scale = self.current_scale()

        if self.map_mode == "body":
            self.draw_body_view()
        elif self.map_mode == "wiki":
            self.draw_wiki_view()
        elif self.map_mode == "system":
            self.draw_system_view(scale)
            if (
                self.scope_mode == "local"
                and self.focus_body
                and any(child for child in self.focus_body.children if not child.is_belt())
            ):
                self.draw_moon_panel(self.focus_body)
        else:
            self.draw_cluster_view(scale)

        self.draw_info_panel()
        
        # Draw UI buttons on top of everything
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.ui_buttons:
            btn.hover = btn.contains(mouse_pos)
            btn.draw(self.screen)
        
        self.draw_help_overlay()
        pygame.display.flip()

    def draw_system_view(self, scale: float):
        # Orbits
        for b in self.system.bodies.values():
            if b.is_root():
                continue
            if b.is_belt() and not self.show_belts:
                continue
            if b.is_moon() and not self.show_moons:
                continue
            if b.type == "dwarf_planet" and not self.show_dwarfs:
                continue
            color = (90, 90, 90) if b.is_belt() else (120, 100, 50)
            self.draw_orbit_for_body(b, scale, color)

        # Bodies
        for b in self.system.bodies.values():
            if b.is_belt():
                continue
            if b.is_moon() and not self.show_moons:
                continue
            if b.type == "dwarf_planet" and not self.show_dwarfs:
                continue
            sx, sy = self.world_to_screen(b.pos[0], b.pos[1], scale)
            size = b.visual_size
            if b.is_moon():
                size = max(size, 3)
            if b.type in ("star", "primary_star"):
                size = max(size, 10)
            if b.type == "barycenter":
                size = 4
            pygame.draw.circle(self.screen, b.color, (int(sx), int(sy)), size)
            if b is self.focus_body:
                pygame.draw.circle(
                    self.screen, (255, 255, 255), (int(sx), int(sy)), size + 4, 1
                )

        # Measurement line (system space)
        if (
            self.measure_mode
            and self.measure_start_world is not None
            and self.measure_end_world is not None
        ):
            ax, ay = self.measure_start_world
            bx, by = self.measure_end_world
            sx1, sy1 = self.world_to_screen(ax, ay, scale)
            sx2, sy2 = self.world_to_screen(bx, by, scale)
            pygame.draw.line(
                self.screen,
                (200, 220, 255),
                (int(sx1), int(sy1)),
                (int(sx2), int(sy2)),
                2,
            )

    def draw_orbit_for_body(self, b: Body, scale: float, color):
        parent = b.parent
        if not parent:
            return
        if self.view_mode == "top":
            cx, cy = self.world_to_screen(parent.pos[0], parent.pos[1], scale)
            r = max(1, int(b.a_au * scale))
            if r > 1:
                pygame.draw.circle(self.screen, color, (int(cx), int(cy)), r, 1)
        else:
            steps = 72
            pts = []
            for i in range(steps + 1):
                theta = 2.0 * math.pi * i / steps
                wx = parent.pos[0] + b.a_au * math.cos(theta)
                wy = parent.pos[1] + b.a_au * math.sin(theta)
                sx, sy = self.world_to_screen(wx, wy, scale)
                pts.append((int(sx), int(sy)))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, color, False, pts, 1)

    def draw_moon_panel(self, center_body: Body):
        moons = [m for m in center_body.children if not m.is_belt()]
        if not moons:
            return

        sidebar_width = int(self.width * 0.32)
        sim_width = self.width - sidebar_width

        panel_width = int(sim_width * 0.55)
        panel_height = int(self.height * 0.30)
        rect = pygame.Rect(
            10,
            self.height - panel_height - 10,
            panel_width,
            panel_height,
        )

        pygame.draw.rect(self.screen, (8, 8, 20), rect)
        pygame.draw.rect(self.screen, (80, 80, 140), rect, 1)

        max_a = max(m.a_au for m in moons) or 1.0
        local_scale = 0.40 * min(rect.width, rect.height) / max_a
        cx = rect.centerx
        cy = rect.centery

        orbit_color = (90, 90, 120)
        moon_hits = []

        for m in moons:
            r = int(m.a_au * local_scale)
            if r <= 1:
                continue
            pygame.draw.circle(self.screen, orbit_color, (cx, cy), r, 1)

        pygame.draw.circle(self.screen, center_body.color, (cx, cy), 6)

        for m in moons:
            x = cx + math.cos(m.angle) * m.a_au * local_scale
            y = cy + math.sin(m.angle) * m.a_au * local_scale
            sx = int(x)
            sy = int(y)
            pygame.draw.circle(self.screen, m.color, (sx, sy), 3)
            moon_hits.append((m, sx, sy))

        label = f"{center_body.name} moon system"
        txt = self.small_font.render(label, True, (210, 210, 230))
        self.screen.blit(txt, (rect.x + 6, rect.y + 6))

        self.moon_panel_state = {
            "rect": rect,
            "center": (cx, cy),
            "center_body": center_body,
            "moons": moon_hits,
        }

    def draw_body_view(self):
        """Draw detailed view of a single body."""
        if not self.focus_body:
            txt = self.font.render("No body selected", True, (200, 200, 200))
            self.screen.blit(txt, (20, 20))
            return

        body = self.focus_body
        
        # Title
        title = self.title_font.render(f"{body.name.upper()} — BODY DETAILS", True, (200, 180, 255))
        self.screen.blit(title, (20, 20))
        
        # Large visual representation
        center_x = self.width // 3
        center_y = self.height // 3
        visual_size = min(80, max(20, body.visual_size * 8))
        pygame.draw.circle(self.screen, body.color, (center_x, center_y), int(visual_size))
        
        # Info panel
        info_x = 40
        info_y = center_y + visual_size + 40
        line_height = 25
        
        info_lines = [
            f"Type: {body.type.replace('_', ' ').title()}",
            f"Radius: {body.radius_km:,.0f} km" if body.radius_km > 0 else "Radius: Unknown",
        ]
        
        if body.parent:
            info_lines.append(f"Parent: {body.parent.name}")
            info_lines.append(f"Semi-major Axis: {body.a_au:.4f} AU")
            info_lines.append(f"Eccentricity: {body.e:.4f}")
            info_lines.append(f"Inclination: {body.inclination_deg:.2f}°")
            
            # Show orbital period from data
            if body.period_years > 0:
                if body.period_years < 0.01:  # Less than ~3.65 days
                    period_days = body.period_years * 365.25
                    info_lines.append(f"Orbital Period: {period_days*24:.2f} hours")
                elif body.period_years < 1:
                    period_days = body.period_years * 365.25
                    info_lines.append(f"Orbital Period: {period_days:.2f} days")
                else:
                    info_lines.append(f"Orbital Period: {body.period_years:.2f} years")
        
        if body.children:
            non_belt_children = [c for c in body.children if not c.is_belt()]
            if non_belt_children:
                info_lines.append(f"Satellites: {len(non_belt_children)}")
        
        for i, line in enumerate(info_lines):
            txt = self.font.render(line, True, (210, 210, 230))
            self.screen.blit(txt, (info_x, info_y + i * line_height))

    def draw_wiki_view(self):
        """Draw wiki/lore view for the focused body or current system."""
        # Title
        title = self.title_font.render("WIKI / LORE DATABASE", True, (200, 180, 255))
        self.screen.blit(title, (20, 20))
        
        # Show focused body info if available, otherwise system info
        if self.focus_body:
            self._draw_wiki_body(self.focus_body)
        else:
            self._draw_wiki_system()
    
    def _draw_wiki_body(self, body):
        """Draw wiki information for a specific body."""
        y = 70
        line_height = 25
        
        # Body name header
        name_txt = self.title_font.render(body.name, True, (255, 220, 150))
        self.screen.blit(name_txt, (40, y))
        y += 40
        
        # Type and basic info
        type_str = body.type.replace('_', ' ').title()
        txt = self.font.render(f"Classification: {type_str}", True, (210, 210, 230))
        self.screen.blit(txt, (40, y))
        y += line_height
        
        if body.parent:
            txt = self.font.render(f"Orbits: {body.parent.name}", True, (210, 210, 230))
            self.screen.blit(txt, (40, y))
            y += line_height
        
        # Meta/lore data
        meta = body.meta or {}
        y += 10
        
        # Display all meta fields in a structured way
        if meta:
            lore_title = self.font.render("═══ PLANETARY DATA ═══", True, (180, 180, 220))
            self.screen.blit(lore_title, (40, y))
            y += line_height + 5
            
            # Physical properties
            if 'gravity_g' in meta:
                txt = self.font.render(f"Surface Gravity: {meta['gravity_g']} g", True, (200, 200, 200))
                self.screen.blit(txt, (60, y))
                y += line_height
            
            if 'atmosphere' in meta:
                txt = self.font.render(f"Atmosphere: {meta['atmosphere']}", True, (200, 200, 200))
                self.screen.blit(txt, (60, y))
                y += line_height
            
            if 'climate' in meta:
                txt = self.font.render(f"Climate: {meta['climate']}", True, (200, 200, 200))
                self.screen.blit(txt, (60, y))
                y += line_height
            
            # Population
            pop = meta.get('population') or meta.get('population_estimate')
            if pop:
                txt = self.font.render(f"Population: {pop}", True, (200, 200, 200))
                self.screen.blit(txt, (60, y))
                y += line_height
            
            # Species
            species = meta.get('primary_species') or meta.get('species')
            if species:
                if isinstance(species, list):
                    species_str = ", ".join(species)
                else:
                    species_str = str(species)
                txt = self.font.render(f"Species: {species_str}", True, (200, 200, 200))
                self.screen.blit(txt, (60, y))
                y += line_height
            
            y += 15
            
            # Description
            desc = meta.get('short_description') or meta.get('description')
            if desc:
                desc_title = self.font.render("═══ DESCRIPTION ═══", True, (180, 180, 220))
                self.screen.blit(desc_title, (40, y))
                y += line_height + 5
                
                # Word wrap the description
                words = desc.split()
                line = ""
                max_width = self.width - 120
                for word in words:
                    test_line = line + word + " "
                    test_txt = self.small_font.render(test_line, True, (200, 200, 200))
                    if test_txt.get_width() > max_width:
                        if line:
                            txt = self.small_font.render(line, True, (200, 200, 200))
                            self.screen.blit(txt, (60, y))
                            y += line_height
                        line = word + " "
                    else:
                        line = test_line
                if line:
                    txt = self.small_font.render(line, True, (200, 200, 200))
                    self.screen.blit(txt, (60, y))
                    y += line_height
            
            # Lore/additional fields
            lore = meta.get('lore')
            if lore:
                y += 10
                lore_title = self.font.render("═══ LORE ═══", True, (180, 180, 220))
                self.screen.blit(lore_title, (40, y))
                y += line_height + 5
                
                # Word wrap lore text
                words = lore.split()
                line = ""
                max_width = self.width - 120
                for word in words:
                    test_line = line + word + " "
                    test_txt = self.small_font.render(test_line, True, (200, 200, 200))
                    if test_txt.get_width() > max_width:
                        if line:
                            txt = self.small_font.render(line, True, (200, 200, 200))
                            self.screen.blit(txt, (60, y))
                            y += line_height
                        line = word + " "
                    else:
                        line = test_line
                if line:
                    txt = self.small_font.render(line, True, (200, 200, 200))
                    self.screen.blit(txt, (60, y))
                    y += line_height
        else:
            txt = self.small_font.render("No wiki data available for this body.", True, (150, 150, 150))
            self.screen.blit(txt, (60, y))
        
        # Tags
        if body.tags:
            y += 20
            tags_str = "Tags: " + ", ".join(body.tags)
            txt = self.small_font.render(tags_str, True, (150, 180, 200))
            self.screen.blit(txt, (40, y))
    
    def _draw_wiki_system(self):
        """Draw wiki information for the current system."""
        y = 70
        line_height = 25
        
        sys_name = getattr(self.system, 'name', 'Unknown System')
        sys_id = getattr(self.system, 'id', '?')
        
        # System name header
        name_txt = self.title_font.render(f"{sys_name} System", True, (255, 220, 150))
        self.screen.blit(name_txt, (40, y))
        y += 40
        
        txt = self.font.render(f"System ID: {sys_id}", True, (210, 210, 230))
        self.screen.blit(txt, (40, y))
        y += line_height + 20
        
        # List all bodies in the system
        lore_title = self.font.render("═══ SYSTEM BODIES ═══", True, (180, 180, 220))
        self.screen.blit(lore_title, (40, y))
        y += line_height + 5
        
        if hasattr(self.system, 'bodies'):
            for body in self.system.bodies.values():
                if not body.is_belt():
                    type_str = body.type.replace('_', ' ').title()
                    body_line = f"• {body.name} ({type_str})"
                    txt = self.small_font.render(body_line, True, (200, 200, 200))
                    self.screen.blit(txt, (60, y))
                    y += line_height
                    
                    # Show short desc if available
                    if body.meta and 'short_description' in body.meta:
                        desc = body.meta['short_description']
                        if len(desc) > 80:
                            desc = desc[:77] + "..."
                        txt = self.small_font.render(f"  {desc}", True, (150, 150, 150))
                        self.screen.blit(txt, (80, y))
                        y += line_height
        else:
            txt = self.small_font.render("No system data available.", True, (150, 150, 150))
            self.screen.blit(txt, (60, y))

    def draw_cluster_view(self, scale: float):
        """Draw star cluster: each system as a point in offset space."""
        if not self.system_defs:
            txt = self.font.render(
                "No universe systems defined for cluster view.", True, (220, 220, 220)
            )
            self.screen.blit(txt, (20, 20))
            return

        # Draw title
        title = self.title_font.render("STAR CLUSTER MAP", True, (200, 180, 255))
        self.screen.blit(title, (20, 20))
        
        instruction = self.small_font.render("Click on any star to explore • Use scroll to zoom • Drag to pan", True, (150, 150, 200))
        self.screen.blit(instruction, (20, 50))

        # Get mouse position for hover detection
        mouse_pos = pygame.mouse.get_pos()
        hovered_system = None
        hovered_idx = None
        
        # Check which system is being hovered
        for idx, sd in enumerate(self.system_defs):
            ox, oy = sd["offset"]
            sx, sy = self.world_to_screen(ox, oy, scale)
            dx = sx - mouse_pos[0]
            dy = sy - mouse_pos[1]
            d2 = dx * dx + dy * dy
            if d2 <= (18 * 18):  # Hit radius
                hovered_system = sd["id"]
                hovered_idx = idx
                break
        
        # Store hovered index for button callbacks
        self.hovered_system_idx = hovered_idx

        # Draw system nodes
        for idx, sd in enumerate(self.system_defs):
            sid = sd["id"]
            name = sd["name"]
            ox, oy = sd["offset"]
            color = sd["color"]

            sx, sy = self.world_to_screen(ox, oy, scale)
            
            # Larger radius for easier clicking
            r = 10
            
            # Current system: bright white ring
            if sid == self.current_system_id:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(sx), int(sy)), r + 5, 2)
            
            # Hovered system: bright glow
            if sid == hovered_system:
                pygame.draw.circle(self.screen, (255, 200, 100), (int(sx), int(sy)), r + 8, 1)
                pygame.draw.circle(self.screen, (255, 150, 50), (int(sx), int(sy)), r + 5, 1)
            
            # System body
            pygame.draw.circle(self.screen, color, (int(sx), int(sy)), r)
            
            # Name label
            txt = self.small_font.render(name, True, (210, 210, 230))
            self.screen.blit(txt, (int(sx) + 14, int(sy) - 6))

        # Measurement line in cluster space
        if (
            self.measure_mode
            and self.measure_start_world is not None
            and self.measure_end_world is not None
        ):
            ax, ay = self.measure_start_world
            bx, by = self.measure_end_world
            sx1, sy1 = self.world_to_screen(ax, ay, scale)
            sx2, sy2 = self.world_to_screen(bx, by, scale)
            pygame.draw.line(
                self.screen,
                (200, 220, 255),
                (int(sx1), int(sy1)),
                (int(sx2), int(sy2)),
                2,
            )

    def draw_info_panel(self):
        sidebar_width = int(self.width * 0.32)
        rect = pygame.Rect(self.width - sidebar_width, 0, sidebar_width, self.height)
        pygame.draw.rect(self.screen, (5, 5, 15), rect)
        pygame.draw.rect(self.screen, (60, 60, 90), rect, 1)

        lines: List[str] = []

        # System header
        sys_id = getattr(self.system, "id", "?")
        sys_name = getattr(self.system, "name", sys_id)
        if self.system_defs:
            ids = [sd["id"] for sd in self.system_defs]
            try:
                idx = ids.index(self.current_system_id)
            except ValueError:
                idx = 0
            total = len(self.system_defs)
            lines.append(f"System [{idx+1}/{total}]: {sys_name} ({sys_id})")
        else:
            lines.append(f"System: {sys_name} ({sys_id})")

        lines.append(
            f"Map: {self.map_mode:<7}   View: {self.view_mode:<3}   Scope: {self.scope_mode:<6}"
        )
        lines.append(
            f"Speed x{self.speed_multiplier:.3f}   Follow: {'on' if self.follow_focus else 'off'}"
        )

        if self.map_mode == "system":
            b = self.focus_body
            if b and b.period_years > 0:
                seconds_per_orbit = BASE_ORBIT_SECONDS / self.speed_multiplier
                lines.append(f"~{seconds_per_orbit:.1f}s per orbit at this speed")
            lines.append(f"Sim time: {self.sim_time_years:8.3f} years")
        else:
            lines.append("Sim time: (cluster view — static star positions)")

        # Focus block (system mode)
        if self.map_mode == "system":
            b = self.focus_body
            if b:
                lines.append(f"Focus: {b.name} [{b.type}]")
                if b.a_au > 0:
                    period = b.period_years
                    a_km = b.a_au * AU_IN_KM
                    circumference = 2.0 * math.pi * a_km
                    ship_time_sec = circumference / SHIP_SPEED_KM_S
                    ship_days = ship_time_sec / 86400.0
                    lines.append(f"a = {b.a_au:.3f} AU   T = {period:.3f} years")
                    lines.append(
                        f"Orbit ~ {circumference:,.0f} km; {ship_days:,.1f} d @ {SHIP_SPEED_KM_S:.0f} km/s"
                    )
                    light_seconds = a_km / SPEED_OF_LIGHT_KM_S
                    light_minutes = light_seconds / 60.0
                    lines.append(f"≈ {light_minutes:.1f} light-minutes from center")
                else:
                    lines.append("a = 0 (central body)")

                meta = b.meta or {}
                g = meta.get("gravity_g")
                radius_meta = meta.get("radius_km")
                radius = radius_meta if radius_meta is not None else (b.radius_km or None)
                pop = meta.get("population") or meta.get("population_estimate")

                stats_bits = []
                if g is not None:
                    stats_bits.append(f"g ≈ {g}")
                if radius is not None:
                    try:
                        stats_bits.append(f"R ≈ {float(radius):,.0f} km")
                    except Exception:
                        stats_bits.append(f"R ≈ {radius} km")
                if pop is not None:
                    stats_bits.append(f"pop ~ {pop}")
                if stats_bits:
                    lines.append(" / ".join(stats_bits))

                atm = meta.get("atmosphere")
                climate = meta.get("climate")
                env_bits = []
                if atm:
                    env_bits.append(f"atm: {atm}")
                if climate:
                    env_bits.append(f"climate: {climate}")
                if env_bits:
                    lines.append(" ; ".join(env_bits))

                species = meta.get("primary_species") or meta.get("species")
                if isinstance(species, list):
                    species_str = ", ".join(species)
                else:
                    species_str = species
                if species_str:
                    lines.append(f"species: {species_str}")

                desc = meta.get("short_description")
                if desc:
                    lines.append("")
                    lines.append(desc)

        # Measurement block
        lines.append("")
        if self.map_mode == "system":
            lines.append(f"Measure mode: {'on' if self.measure_mode else 'off'} (AU)")
        else:
            lines.append(f"Measure mode: {'on' if self.measure_mode else 'off'} (ly)")

        if (
            self.measure_mode
            and self.measure_start_world is not None
            and self.measure_end_world is not None
        ):
            ax, ay = self.measure_start_world
            bx, by = self.measure_end_world
            dx = bx - ax
            dy = by - ay
            raw_dist = math.hypot(dx, dy)

            if raw_dist > 1e-6:
                if self.map_mode == "system":
                    dist_au = raw_dist
                    dist_km = dist_au * AU_IN_KM
                    lines.append(f"d ≈ {dist_au:.3f} AU (~{dist_km:,.0f} km)")
                else:
                    dist_ly = raw_dist
                    dist_km = dist_ly * LY_IN_KM
                    lines.append(f"d ≈ {dist_ly:.3f} ly (~{dist_km:,.0f} km)")

                light_seconds = dist_km / SPEED_OF_LIGHT_KM_S
                if light_seconds < 60:
                    lines.append(f"light-time ≈ {light_seconds:.1f} s")
                elif light_seconds < 3600:
                    lines.append(f"light-time ≈ {light_seconds/60.0:.1f} min")
                elif light_seconds < 86400:
                    lines.append(f"light-time ≈ {light_seconds/3600.0:.1f} h")
                else:
                    lines.append(f"light-time ≈ {light_seconds/86400.0:.2f} d")

                for frac in (0.1, 0.01):
                    v = SPEED_OF_LIGHT_KM_S * frac
                    t_sec = dist_km / v
                    t_days = t_sec / 86400.0
                    lines.append(f"@ {frac:.0%} c: {t_days:.2f} d")

                t_ship_sec = dist_km / SHIP_SPEED_KM_S
                t_ship_years = t_ship_sec / (86400.0 * 365.25)
                lines.append(f"@ {SHIP_SPEED_KM_S:.0f} km/s: {t_ship_years:.2f} yr")
            else:
                lines.append("Drag to measure a distance vector.")
        else:
            if self.measure_mode:
                lines.append("Drag to measure a distance vector.")

        # Render
        x = rect.x + 10
        y = rect.y + 10
        for line in lines[:28]:
            txt = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(txt, (x, y))
            y += txt.get_height() + 2

    def draw_help_overlay(self):
        if self.map_mode == "cluster":
            help_lines = [
                "CLUSTER VIEW: Click any star system to explore it   G: return to system view   Zoom & pan: same as system",
                "SPACE: play/pause   , / .: prev/next system   +/- or wheel: zoom   0: reset zoom+pan   LEFT drag: pan",
            ]
        else:
            help_lines = [
                "SPACE: play/pause   [ / ]: slower/faster   +/- or wheel: zoom   0: reset zoom+pan",
                "TAB/Shift+TAB/↑↓: focus body   LEFT drag: pan   tap: select   1/2/3: belts/moons/dwarfs   V: top/iso",
                "M: moon panel   F: follow focus   D: measure mode (L-drag)   , / .: prev/next system   G: system/cluster view",
            ]
        y = self.height - len(help_lines) * (self.small_font.get_height() + 2) - 4
        for line in help_lines:
            txt = self.small_font.render(line, True, (150, 150, 150))
            self.screen.blit(txt, (10, y))
            y += txt.get_height() + 2


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser(description="Fictional solar system sandbox viewer")
    parser.add_argument("json_path", help="Path to system JSON or universe JSON.")
    parser.add_argument(
        "--system",
        dest="system_id",
        default=None,
        help="System id to load when using a universe.json file.",
    )
    args = parser.parse_args()

    system = load_system_from_json(args.json_path, system_id=args.system_id)
    system.update(0.0)

    system_defs = list_systems_in_universe(args.json_path)
    current_system_id = args.system_id
    if current_system_id is None:
        if system_defs:
            current_system_id = system_defs[0]["id"]
        else:
            current_system_id = getattr(system, "id", "system")

    sandbox = OrbitSandbox(
        system,
        json_path=args.json_path,
        system_defs=system_defs,
        current_system_id=current_system_id,
    )
    sandbox.run()


if __name__ == "__main__":
    main()
