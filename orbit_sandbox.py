#!/usr/bin/env python3
"""
orbit_sandbox.py

Fictional-system orbit sandbox.

Core features
-------------
- Top-down / isometric views (V).
- System view + optional moon mini-panel (M) for focused body.
- Wiki sidebar: meta for ANY focused body (star, planet, moon, station, etc.).
- Orbit-relative time scaling:
    * Focus a body (planet or moon) -> simulation speed is scaled so
      that its orbital period ~ BASE_ORBIT_SECONDS at speed x1.
- Big hitboxes for selecting bodies (dots OR orbit rings).
- SIMPLE interactions:
    * Normal mode (measure OFF):
        - Left-click: select/focus body or orbit.
        - Left-drag: pan camera.
    * Measure mode (D toggles):
        - Left-click+drag: draw a distance segment anywhere in space.
- NEW:
    * Moons in the moon mini-panel are clickable:
        - clicking a moon selects it (wiki + orbit info).
        - clicking near the central globe selects the planet.
    * Camera follow mode (F):
        - When ON, camera stays locked to the focused body’s orbit.
        - When OFF, panning is manual.

Usage
-----
    python orbit_sandbox.py data/system.json
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

        # Orbital period (years) – Kepler-ish if missing
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

        # Lore/meta
        self.tags: List[str] = raw.get("tags", [])
        self.image: Optional[str] = raw.get("image")
        self.meta: dict = raw.get("meta", {})

        # Position in world space (AU)
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

        # link parents/children
        for b in self.bodies.values():
            if b.parent_id:
                parent = self.bodies.get(b.parent_id)
                if parent:
                    b.parent = parent
                    parent.children.append(b)

        self.roots: List[Body] = [b for b in self.bodies.values() if b.is_root()]
        if not self.roots:
            raise ValueError("System must have at least one root (e.g., barycenter).")

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


# ---------- Loading JSON ----------


def load_system_from_json(path: str, system_id: Optional[str] = None) -> SystemModel:
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

    sid = data.get("id", "system")
    name = data.get("name", sid)
    bodies_data = data["bodies"]
    return SystemModel(sid, name, bodies_data)


# ---------- Viewer / UI ----------


class OrbitSandbox:
    def __init__(self, system: SystemModel, width: int = 1200, height: int = 700):
        pygame.init()
        pygame.display.set_caption("orbit_sandbox")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.system = system
        self.sim_time_years = 0.0
        self.running = True

        # Orbit-relative time scale
        self.base_time_scale = 1.0
        self.speed_multiplier = 1.0
        self.time_scale = 1.0

        # Views
        self.view_mode = "top"      # "top" or "iso"
        self.scope_mode = "system"  # "system" or "local"

        # Camera center (world coords, AU)
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.follow_focus = False   # F toggles camera follow

        # Zoom
        self.base_pixels_per_au = 0.44 * min(self.width, self.height) / system.max_a_au
        self.zoom = 1.0

        # Fonts
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)

        # Visibility toggles
        self.show_belts = True
        self.show_moons = True
        self.show_dwarfs = True

        # Focus list
        self.focusables = system.ordered_focusable_bodies()
        self.focus_index = 0 if self.focusables else -1
        self.focus_body: Optional[Body] = (
            self.focusables[self.focus_index] if self.focus_index >= 0 else None
        )

        if self.focus_body:
            self.cam_x, self.cam_y = self.focus_body.pos

        # Drag / pan (left button)
        self.left_down = False
        self.left_down_pos: Tuple[int, int] = (0, 0)
        self.pan_active = False
        self.pan_anchor_world: Optional[Tuple[float, float]] = None
        self.drag_threshold_px = 5  # <= this = click, > this = pan

        # Measurement
        self.measure_mode = False
        self.measure_dragging = False
        self.measure_start_world: Optional[Tuple[float, float]] = None
        self.measure_end_world: Optional[Tuple[float, float]] = None

        # Moon panel hit detection
        self.moon_panel_rect: Optional[pygame.Rect] = None
        self.moon_panel_center: Optional[Tuple[int, int]] = None
        self.moon_panel_scale: float = 1.0
        self.moon_panel_parent: Optional[Body] = None
        self.moon_panel_moons: List[Body] = []

        self.recalc_time_scale()

    # --- time scaling ---

    def recalc_time_scale(self):
        """Set time_scale so focused body's orbit ~ BASE_ORBIT_SECONDS at speed x1."""
        b = self.focus_body
        if b and b.period_years > 0:
            self.base_time_scale = b.period_years / BASE_ORBIT_SECONDS
        else:
            self.base_time_scale = 1.0
        self.time_scale = self.base_time_scale * self.speed_multiplier

    # --- camera helpers ---

    def camera_center(self) -> Tuple[float, float]:
        return self.cam_x, self.cam_y

    # --- transforms ---

    def world_to_screen(self, wx: float, wy: float, scale: float) -> Tuple[float, float]:
        cam_x, cam_y = self.camera_center()
        dx = wx - cam_x
        dy = wy - cam_y

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
        cam_x, cam_y = self.camera_center()
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
        wx = cam_x + dx
        wy = cam_y + dy
        return wx, wy

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

    # --- main loop ---

    def run(self):
        while True:
            dt_real = self.clock.tick(60) / 1000.0
            self.handle_events()
            if self.running:
                dt_years = dt_real * self.time_scale
                self.sim_time_years += dt_years
                self.system.update(dt_years)
                if self.follow_focus and self.focus_body:
                    # Lock camera to focused body's orbit
                    self.cam_x, self.cam_y = self.focus_body.pos
            self.draw()

    # --- event handling ---

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

            if event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_left_down(event.pos)

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.handle_left_up(event.pos)

            if event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event.pos)

            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.zoom *= 1.1
                elif event.y < 0:
                    self.zoom /= 1.1
                self.zoom = max(0.1, min(self.zoom, 10.0))

    def handle_keydown(self, key):
        if key == pygame.K_SPACE:
            self.running = not self.running

        # focus movement
        if key == pygame.K_UP:
            self.cycle_focus(1)
        if key == pygame.K_DOWN:
            self.cycle_focus(-1)

        # zoom
        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.zoom *= 1.1
        if key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.zoom /= 1.1
        self.zoom = max(0.1, min(self.zoom, 10.0))

        # time scale multiplier (global)
        if key == pygame.K_RIGHTBRACKET:  # ]
            self.speed_multiplier *= 2.0
            self.speed_multiplier = min(self.speed_multiplier, 128.0)
            self.recalc_time_scale()
        if key == pygame.K_LEFTBRACKET:  # [
            self.speed_multiplier /= 2.0
            self.speed_multiplier = max(self.speed_multiplier, 1.0 / 128.0)
            self.recalc_time_scale()

        if key == pygame.K_0:
            self.zoom = 1.0

        # Recenter on focus
        if key == pygame.K_c and self.focus_body:
            self.cam_x, self.cam_y = self.focus_body.pos

        # Focus cycling
        if key == pygame.K_TAB:
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.cycle_focus(-1)
            else:
                self.cycle_focus(1)

        # Toggles
        if key == pygame.K_1:
            self.show_belts = not self.show_belts
        if key == pygame.K_2:
            self.show_moons = not self.show_moons
        if key == pygame.K_3:
            self.show_dwarfs = not self.show_dwarfs

        # View mode
        if key == pygame.K_v:
            self.view_mode = "iso" if self.view_mode == "top" else "top"

        # Scope mode (moon panel on/off)
        if key == pygame.K_m:
            if self.scope_mode == "system":
                if self.focus_body and self.focus_body.children:
                    self.scope_mode = "local"
            else:
                self.scope_mode = "system"

        # Measure mode (D)
        if key == pygame.K_d:
            self.measure_mode = not self.measure_mode
            self.measure_dragging = False
            self.measure_start_world = None
            self.measure_end_world = None

        # Follow focus (F)
        if key == pygame.K_f:
            self.follow_focus = not self.follow_focus

    # --- mouse handlers ---

    def handle_left_down(self, pos):
        mx, my = pos
        scale = self.base_pixels_per_au * self.zoom

        self.left_down = True
        self.left_down_pos = (mx, my)
        self.pan_active = False

        if self.measure_mode:
            # Start measurement drag
            wx, wy = self.screen_to_world(mx, my, scale)
            self.measure_start_world = (wx, wy)
            self.measure_end_world = (wx, wy)
            self.measure_dragging = True
        else:
            # Potential click or pan; record world anchor
            wx, wy = self.screen_to_world(mx, my, scale)
            self.pan_anchor_world = (wx, wy)

    def handle_left_up(self, pos):
        mx, my = pos
        scale = self.base_pixels_per_au * self.zoom

        if self.measure_mode:
            if self.measure_dragging:
                wx, wy = self.screen_to_world(mx, my, scale)
                self.measure_end_world = (wx, wy)
                self.measure_dragging = False
        else:
            # If we didn't pan, treat as click
            dx = mx - self.left_down_pos[0]
            dy = my - self.left_down_pos[1]
            if not self.pan_active and (dx * dx + dy * dy) <= self.drag_threshold_px ** 2:
                self.handle_click(mx, my)

        self.left_down = False
        self.pan_active = False

    def handle_mouse_motion(self, pos):
        mx, my = pos
        scale = self.base_pixels_per_au * self.zoom

        # Measurement drag always honored, even if following
        if self.measure_mode and self.measure_dragging:
            wx, wy = self.screen_to_world(mx, my, scale)
            self.measure_end_world = (wx, wy)
            return

        # If we're following focus, don't pan at all
        if self.follow_focus:
            return

        # Panning in normal mode
        if self.left_down and not self.measure_mode and self.pan_anchor_world is not None:
            dx = mx - self.left_down_pos[0]
            dy = my - self.left_down_pos[1]

            if not self.pan_active:
                if dx * dx + dy * dy > self.drag_threshold_px ** 2:
                    self.pan_active = True

            if self.pan_active:
                wx, wy = self.pan_anchor_world
                if self.view_mode == "top":
                    dx_world = (mx - self.width / 2) / scale
                    dy_world = (my - self.height / 2) / scale
                    self.cam_x = wx - dx_world
                    self.cam_y = wy - dy_world
                else:
                    X = (mx - self.width / 2) / scale
                    Y = (my - self.height / 2) / scale
                    dx_minus_dy = X / 0.75
                    dx_plus_dy = Y / 0.40
                    dx_iso = (dx_minus_dy + dx_plus_dy) / 2.0
                    dy_iso = (dx_plus_dy - dx_minus_dy) / 2.0
                    self.cam_x = wx - dx_iso
                    self.cam_y = wy - dy_iso

    # --- clicking / focus ---

    def handle_click(self, mx: int, my: int):
        # First: if we're in local scope and click inside the moon panel, use that
        if (
            self.scope_mode == "local"
            and self.moon_panel_rect is not None
            and self.moon_panel_rect.collidepoint(mx, my)
        ):
            moon = self.pick_moonpanel_body_at(mx, my)
            if moon:
                self.set_focus(moon)
                return
            # Click near center selects parent planet
            if self.moon_panel_center and self.moon_panel_parent:
                cx, cy = self.moon_panel_center
                if (mx - cx) ** 2 + (my - cy) ** 2 <= 12 ** 2:
                    self.set_focus(self.moon_panel_parent)
                    return

        # Otherwise: normal system view picking
        body = self.pick_body_at(mx, my)
        if body:
            self.set_focus(body)

    def pick_body_at(self, mx: float, my: float) -> Optional[Body]:
        scale = self.base_pixels_per_au * self.zoom

        best_body = None
        best_score = float("inf")

        for b in self.system.bodies.values():
            if b.is_belt():
                continue

            sx, sy = self.world_to_screen(b.pos[0], b.pos[1], scale)
            dx = sx - mx
            dy = sy - my
            d2 = dx * dx + dy * dy
            body_r = max(14, b.visual_size + 10)
            candidate_score = float("inf")

            if d2 <= body_r * body_r:
                candidate_score = d2

            if b.parent is not None and b.a_au > 0:
                px, py = self.world_to_screen(b.parent.pos[0], b.parent.pos[1], scale)
                d_click_parent = math.hypot(mx - px, my - py)
                r_orbit = b.a_au * scale
                if r_orbit > 4:
                    diff = abs(d_click_parent - r_orbit)
                    if diff < 10.0:
                        candidate_score = min(candidate_score, diff * diff)

            if candidate_score < best_score:
                best_score = candidate_score
                best_body = b

        if best_score == float("inf"):
            return None
        return best_body

    def pick_moonpanel_body_at(self, mx: float, my: float) -> Optional[Body]:
        """Pick a moon from the mini-panel (screen coords)."""
        if (
            self.moon_panel_rect is None
            or self.moon_panel_center is None
            or not self.moon_panel_moons
        ):
            return None

        cx, cy = self.moon_panel_center
        scale = self.moon_panel_scale

        best_moon = None
        best_score = float("inf")
        for m in self.moon_panel_moons:
            x = cx + math.cos(m.angle) * m.a_au * scale
            y = cy + math.sin(m.angle) * m.a_au * scale
            dx = mx - x
            dy = my - y
            d2 = dx * dx + dy * dy
            if d2 < best_score and d2 <= 10 ** 2:
                best_score = d2
                best_moon = m

        return best_moon

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

    # --- drawing ---

    def draw(self):
        self.screen.fill((0, 0, 0))

        # reset moon-panel hit info for this frame
        self.moon_panel_rect = None
        self.moon_panel_center = None
        self.moon_panel_parent = None
        self.moon_panel_moons = []
        self.moon_panel_scale = 1.0

        self.draw_system_view()

        if (
            self.scope_mode == "local"
            and self.focus_body
            and any(child for child in self.focus_body.children if not child.is_belt())
        ):
            self.draw_moon_panel(self.focus_body)

        if self.measure_start_world and self.measure_end_world:
            self.draw_measurement_line()

        self.draw_info_panel()
        self.draw_help_overlay()

        pygame.display.flip()

    def draw_system_view(self):
        scale = self.base_pixels_per_au * self.zoom

        # orbits
        for b in self.system.bodies.values():
            if b.is_root():
                continue
            if b.is_belt() and not self.show_belts:
                continue
            if b.is_moon() and not self.show_moons:
                continue
            if b.type == "dwarf_planet" and not self.show_dwarfs:
                continue

            if b.is_belt():
                col = (90, 90, 90)
            else:
                col = (120, 100, 50)

            self.draw_orbit_for_body(b, scale, col)

        # bodies
        for b in self.system.bodies.values():
            if b.is_belt():
                continue
            if b.is_moon() and not self.show_moons:
                continue
            if b.type == "dwarf_planet" and not self.show_dwarfs:
                continue

            sx, sy = self.world_to_screen(b.pos[0], b.pos[1], scale)

            size = b.visual_size
            if b.type in ("star", "primary_star"):
                size = max(size, 10)
            if b.type == "barycenter":
                size = 4

            pygame.draw.circle(self.screen, b.color, (int(sx), int(sy)), size)

            if b is self.focus_body:
                pygame.draw.circle(
                    self.screen, (255, 255, 255), (int(sx), int(sy)), size + 4, 1
                )

    def draw_moon_panel(self, center_body: Body):
        moons = [m for m in center_body.children if not m.is_belt()]
        if not moons:
            return

        sidebar_width = int(self.width * 0.32)
        sim_width = self.width - sidebar_width

        panel_width = int(sim_width * 0.55)
        panel_height = int(self.height * 0.35)
        panel_x = 10
        panel_y = self.height - panel_height - 10

        rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        self.moon_panel_rect = rect
        self.moon_panel_parent = center_body

        pygame.draw.rect(self.screen, (8, 8, 20), rect)
        pygame.draw.rect(self.screen, (80, 80, 140), rect, 1)

        max_a = max(m.a_au for m in moons) or 1.0
        local_scale = 0.45 * min(rect.width, rect.height) / max_a
        self.moon_panel_scale = local_scale

        cx = rect.centerx
        cy = rect.centery
        self.moon_panel_center = (cx, cy)
        self.moon_panel_moons = moons

        orbit_color = (90, 90, 120)
        for m in moons:
            r = int(m.a_au * local_scale)
            if r <= 1:
                continue
            pygame.draw.circle(self.screen, orbit_color, (cx, cy), r, 1)

        pygame.draw.circle(self.screen, center_body.color, (cx, cy), 8)

        for m in moons:
            x = cx + math.cos(m.angle) * m.a_au * local_scale
            y = cy + math.sin(m.angle) * m.a_au * local_scale
            pygame.draw.circle(self.screen, m.color, (int(x), int(y)), 4)

        label = f"{center_body.name} moon system"
        txt = self.small_font.render(label, True, (210, 210, 230))
        self.screen.blit(txt, (rect.x + 5, rect.y + 5))

    def draw_measurement_line(self):
        if not (self.measure_start_world and self.measure_end_world):
            return

        scale = self.base_pixels_per_au * self.zoom
        ax, ay = self.measure_start_world
        bx, by = self.measure_end_world

        sx1, sy1 = self.world_to_screen(ax, ay, scale)
        sx2, sy2 = self.world_to_screen(bx, by, scale)

        pygame.draw.line(
            self.screen, (0, 200, 255), (int(sx1), int(sy1)), (int(sx2), int(sy2)), 2
        )
        pygame.draw.circle(self.screen, (0, 200, 255), (int(sx1), int(sy1)), 3, 1)
        pygame.draw.circle(self.screen, (0, 200, 255), (int(sx2), int(sy2)), 3, 1)

    def draw_info_panel(self):
        sidebar_width = int(self.width * 0.32)
        rect = pygame.Rect(self.width - sidebar_width, 0, sidebar_width, self.height)

        pygame.draw.rect(self.screen, (5, 5, 15), rect)
        pygame.draw.rect(self.screen, (60, 60, 90), rect, 1)

        lines: List[str] = []

        lines.append(f"System: {self.system.name}")
        lines.append(
            f"View: {self.view_mode:<3}   Scope: {self.scope_mode:<6}   Measure: {'ON' if self.measure_mode else 'off'}   Follow: {'ON' if self.follow_focus else 'off'}"
        )
        lines.append(f"Speed x{self.speed_multiplier:.3f}")

        b = self.focus_body
        if b and b.period_years > 0:
            seconds_per_orbit = BASE_ORBIT_SECONDS / self.speed_multiplier
            lines.append(f"~{seconds_per_orbit:.1f}s per orbit for {b.name}")

        lines.append(f"Sim time: {self.sim_time_years:8.3f} years")

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

        if self.measure_start_world and self.measure_end_world:
            ax, ay = self.measure_start_world
            bx, by = self.measure_end_world
            dx = bx - ax
            dy = by - ay
            dist_au = math.hypot(dx, dy)
            dist_km = dist_au * AU_IN_KM

            lines.append("")
            lines.append("Measure segment:")
            lines.append(f"d ≈ {dist_au:.3f} AU (~{dist_km:,.0f} km)")

            if dist_km > 0:
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
                    lines.append(f"@{frac:.0%} c: {t_days:.2f} d")

                t_ship_sec = dist_km / SHIP_SPEED_KM_S
                t_ship_years = t_ship_sec / (86400.0 * 365.25)
                lines.append(f"@{SHIP_SPEED_KM_S:.0f} km/s: {t_ship_years:.2f} yr")

        x = rect.x + 10
        y = rect.y + 10
        for line in lines[:26]:
            txt = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(txt, (x, y))
            y += txt.get_height() + 2

    def draw_help_overlay(self):
        help_lines = [
            "SPACE: play/pause   [ / ]: slower/faster   +/- or wheel: zoom   0: reset zoom   C: recenter",
            "TAB / Shift+TAB / ↑↓: focus   L-click: select   L-drag: pan   M: moon panel   V: top/iso",
            "D: measure (L-drag)   F: follow focus   1/2/3: belts/moons/dwarfs   Click moons in mini-panel to focus",
        ]
        y = self.height - 3 * (self.small_font.get_height() + 2) - 4
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
    sandbox = OrbitSandbox(system)
    sandbox.system.update(0.0)
    sandbox.run()


if __name__ == "__main__":
    main()
