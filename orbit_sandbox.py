#!/usr/bin/env python3
"""
orbit_sandbox.py

Sandbox orbit viewer for fictional systems.

This version:
- Top-down and isometric views (V).
- Orbit-relative time scaling (inner faster, outer slower; uses period_years/a^(3/2)).
- LEFT-DRAG panning in normal mode (no right-click).
- Left-click tap to focus a body (planet, star, moon, or orbit).
- D = measure mode; in measure mode, LEFT-DRAG draws a distance vector.
- Moon mini-panel for focused body with moons (M), with clickable moons.
- Wiki-ish sidebar on the right with orbit stats and meta.
- Optional smooth follow of focused body (F) so you don’t chase it.
- Multi-system nav when using a universe.json:
    * , (comma) = previous system
    * . (period) = next system
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

        # Orbital period (years). If missing but a_au > 0, use Kepler-ish scaling.
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

        # World position in AU
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
            raise ValueError("System must have at least one root.")

        # Largest orbital radius for scaling
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
    """
    Load either a single-system JSON or a universe.json (with 'systems' array)
    and return a SystemModel for the requested system_id (or the first).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Universe form
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
                raise ValueError(f"System id '{system_id}' not found.")
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

    # Single system form
    sid = data.get("id", "system")
    name = data.get("name", sid)
    bodies_data = data["bodies"]
    return SystemModel(sid, name, bodies_data)


def list_systems_in_universe(path: str) -> List[dict]:
    """Return a list of system descriptors [{id, name}] if path is a universe.json, else []."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    systems = []
    if "systems" in data and isinstance(data["systems"], list):
        for s in data["systems"]:
            sid = s.get("id")
            if not sid:
                continue
            systems.append({"id": sid, "name": s.get("name", sid)})
    return systems


# ---------- Viewer / UI ----------


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

        # multi-system context
        self.json_path = json_path
        self.system_defs = system_defs or []
        self.current_system_id = current_system_id

        self.system = system
        self.sim_time_years = 0.0
        self.running = True

        # time scaling
        self.base_time_scale = 1.0
        self.speed_multiplier = 1.0
        self.time_scale = 1.0

        self.view_mode = "top"       # "top" or "iso"
        self.scope_mode = "system"   # "system" or "local" (moon panel)

        self.base_pixels_per_au = 0.44 * min(self.width, self.height) / system.max_a_au
        self.zoom = 1.0

        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)

        self.show_belts = True
        self.show_moons = True
        self.show_dwarfs = True

        self.focusables = system.ordered_focusable_bodies()
        self.focus_index = 0 if self.focusables else -1
        self.focus_body: Optional[Body] = (
            self.focusables[self.focus_index] if self.focus_index >= 0 else None
        )

        # Camera center in world coords
        if self.focus_body:
            self.cam_x, self.cam_y = self.focus_body.pos
        else:
            self.cam_x, self.cam_y = 0.0, 0.0

        # Left-drag for panning
        self.left_dragging = False
        self.left_drag_start_screen: Tuple[int, int] = (0, 0)
        self.left_drag_start_cam: Tuple[float, float] = (self.cam_x, self.cam_y)
        self.left_drag_max_dist2: float = 0.0

        # Measurement via left-drag in measure mode
        self.measure_mode = False
        self.measure_dragging = False
        self.measure_start_world: Optional[Tuple[float, float]] = None
        self.measure_end_world: Optional[Tuple[float, float]] = None

        # Smooth follow toggle
        self.follow_focus = False

        # Moon panel hit-test state
        self.moon_panel_state = None  # dict with rect, center_body, moons[(Body, x, y)]

        self.recalc_time_scale()

    # --- time scaling ---

    def recalc_time_scale(self):
        b = self.focus_body
        if b and b.period_years > 0:
            self.base_time_scale = b.period_years / BASE_ORBIT_SECONDS
        else:
            self.base_time_scale = 1.0
        self.time_scale = self.base_time_scale * self.speed_multiplier

    # --- transforms ---

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
            return self.cam_x + dx, self.cam_y + dy
        else:
            X = (sx - self.width / 2) / scale
            Y = (sy - self.height / 2) / scale
            dx_minus_dy = X / 0.75
            dx_plus_dy = Y / 0.40
            dx = (dx_minus_dy + dx_plus_dy) / 2.0
            dy = (dx_plus_dy - dx_minus_dy) / 2.0
            return self.cam_x + dx, self.cam_y + dy

    # --- main loop ---

    def run(self):
        while True:
            dt_real = self.clock.tick(60) / 1000.0
            self.handle_events()
            if self.running:
                dt_years = dt_real * self.time_scale
                self.sim_time_years += dt_years
                self.system.update(dt_years)
                self.update_camera_follow()
            self.draw()

    # --- camera follow ---

    def update_camera_follow(self):
        if not self.follow_focus:
            return
        if self.left_dragging:  # let manual pan win while dragging
            return
        if not self.focus_body:
            return
        target_x, target_y = self.focus_body.pos
        alpha = 0.12  # smoothing factor
        self.cam_x += (target_x - self.cam_x) * alpha
        self.cam_y += (target_y - self.cam_y) * alpha

    # --- multi-system switching ---

    def switch_system(self, delta: int):
        """Cycle to a different system in universe.json, if available."""
        if not self.system_defs or len(self.system_defs) < 2:
            return

        # Determine current index
        cur_id = self.current_system_id
        if cur_id is None:
            cur_id = self.system_defs[0]["id"]

        idx = 0
        for i, s in enumerate(self.system_defs):
            if s["id"] == cur_id:
                idx = i
                break

        idx = (idx + delta) % len(self.system_defs)
        new_def = self.system_defs[idx]
        new_id = new_def["id"]

        # Load new system fresh
        new_system = load_system_from_json(self.json_path, system_id=new_id)
        new_system.update(0.0)

        self.system = new_system
        self.current_system_id = new_id

        # Recompute scaling
        self.base_pixels_per_au = 0.44 * min(self.width, self.height) / self.system.max_a_au
        self.zoom = 1.0

        # Reset focusables and focus
        self.focusables = self.system.ordered_focusable_bodies()
        self.focus_index = 0 if self.focusables else -1
        self.focus_body = self.focusables[0] if self.focus_index >= 0 else None

        if self.focus_body:
            self.cam_x, self.cam_y = self.focus_body.pos
        else:
            self.cam_x, self.cam_y = 0.0, 0.0

        # Reset time + measurement
        self.sim_time_years = 0.0
        self.measure_mode = False
        self.measure_dragging = False
        self.measure_start_world = None
        self.measure_end_world = None

        self.recalc_time_scale()

    # --- events ---

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.handle_left_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
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
        if key == pygame.K_SPACE:
            self.running = not self.running

        # Focus cycling
        if key == pygame.K_UP:
            self.cycle_focus(1)
        if key == pygame.K_DOWN:
            self.cycle_focus(-1)
        if key == pygame.K_TAB:
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.cycle_focus(-1)
            else:
                self.cycle_focus(1)

        # System cycling (universe.json)
        if key == pygame.K_COMMA:
            self.switch_system(-1)
        if key == pygame.K_PERIOD:
            self.switch_system(1)

        # Zoom
        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.zoom *= 1.1
        if key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.zoom /= 1.1
        self.zoom = max(0.1, min(self.zoom, 10.0))

        # Time scaling
        if key == pygame.K_RIGHTBRACKET:
            self.speed_multiplier *= 2.0
            self.speed_multiplier = min(self.speed_multiplier, 128.0)
            self.recalc_time_scale()
        if key == pygame.K_LEFTBRACKET:
            self.speed_multiplier /= 2.0
            self.speed_multiplier = max(self.speed_multiplier, 1.0 / 128.0)
            self.recalc_time_scale()

        # Reset zoom + pan to focused body
        if key == pygame.K_0:
            self.zoom = 1.0
            if self.focus_body:
                self.cam_x, self.cam_y = self.focus_body.pos
            else:
                self.cam_x, self.cam_y = 0.0, 0.0

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

        # Moon panel scope
        if key == pygame.K_m:
            self.scope_mode = "local" if self.scope_mode == "system" else "system"

        # Measure mode
        if key == pygame.K_d:
            self.measure_mode = not self.measure_mode
            if not self.measure_mode:
                self.measure_dragging = False
                self.measure_start_world = None
                self.measure_end_world = None

        # Follow focus toggle
        if key == pygame.K_f:
            self.follow_focus = not self.follow_focus

    # --- mouse handlers ---

    def handle_left_down(self, pos):
        mx, my = pos
        scale = self.base_pixels_per_au * self.zoom
        if self.measure_mode:
            # Start measuring
            self.measure_dragging = True
            self.measure_start_world = self.screen_to_world(mx, my, scale)
            self.measure_end_world = self.measure_start_world
        else:
            # Start panning candidate
            self.left_dragging = True
            self.left_drag_start_screen = (mx, my)
            self.left_drag_start_cam = (self.cam_x, self.cam_y)
            self.left_drag_max_dist2 = 0.0

    def handle_left_up(self, pos):
        mx, my = pos
        scale = self.base_pixels_per_au * self.zoom

        if self.measure_mode:
            if self.measure_dragging:
                self.measure_dragging = False
                self.measure_end_world = self.screen_to_world(mx, my, scale)
        else:
            if self.left_dragging:
                dx = mx - self.left_drag_start_screen[0]
                dy = my - self.left_drag_start_screen[1]
                d2 = dx * dx + dy * dy
                threshold2 = 16  # <=4 px counts as "click"

                if d2 <= threshold2:
                    # treat as tap -> first try moon panel, then main view
                    if not self.try_moon_panel_click(mx, my):
                        body = self.pick_body_at(mx, my)
                        if body:
                            self.set_focus(body)
                self.left_dragging = False

    def handle_mouse_motion(self, pos):
        mx, my = pos
        scale = self.base_pixels_per_au * self.zoom

        # Update measure drag
        if self.measure_mode and self.measure_dragging and self.measure_start_world:
            self.measure_end_world = self.screen_to_world(mx, my, scale)

        # Update pan drag
        if self.left_dragging and not self.measure_mode:
            dx = mx - self.left_drag_start_screen[0]
            dy = my - self.left_drag_start_screen[1]
            self.left_drag_max_dist2 = max(self.left_drag_max_dist2, dx * dx + dy * dy)
            if scale != 0:
                self.cam_x = self.left_drag_start_cam[0] - dx / scale
                self.cam_y = self.left_drag_start_cam[1] - dy / scale

    # --- picking / focus ---

    def try_moon_panel_click(self, mx: int, my: int) -> bool:
        """If click lands on a moon in the mini-panel, focus it."""
        state = self.moon_panel_state
        if not state:
            return False
        rect = state["rect"]
        if not rect.collidepoint(mx, my):
            return False
        for moon, sx, sy in state["moons"]:
            dx = mx - sx
            dy = my - sy
            if dx * dx + dy * dy <= 10 * 10:  # 10 px radius
                self.set_focus(moon)
                return True
        return False

    def pick_body_at(self, mx: int, my: int) -> Optional[Body]:
        scale = self.base_pixels_per_au * self.zoom
        best_body = None
        best_score = float("inf")

        for b in self.system.bodies.values():
            if b.is_belt():
                continue

            # Body hit
            sx, sy = self.world_to_screen(b.pos[0], b.pos[1], scale)
            dx = sx - mx
            dy = sy - my
            d2 = dx * dx + dy * dy
            body_r = max(14, b.visual_size + 10)
            score = float("inf")
            if d2 <= body_r * body_r:
                score = d2

            # Orbit hit
            if b.parent and b.a_au > 0:
                cx, cy = self.world_to_screen(b.parent.pos[0], b.parent.pos[1], scale)
                odx = mx - cx
                ody = my - cy
                dist_center = math.hypot(odx, ody)
                orbit_radius = b.a_au * scale
                if orbit_radius > 4:
                    diff = abs(dist_center - orbit_radius)
                    if diff < 10:
                        score = min(score, diff * diff)

            if score < best_score:
                best_score = score
                best_body = b

        if best_score == float("inf"):
            return None
        return best_body

    def cycle_focus(self, direction: int):
        if not self.focusables:
            return
        self.focus_index = (self.focus_index + direction) % len(self.focusables)
        self.set_focus(self.focusables[self.focus_index])

    def set_focus(self, body: Body):
        self.focus_body = body
        if body in self.focusables:
            self.focus_index = self.focusables.index(body)
        # Snap camera to the new focus immediately
        self.cam_x, self.cam_y = body.pos
        self.recalc_time_scale()

    # --- drawing ---

    def draw(self):
        self.screen.fill((0, 0, 0))

        # reset moon panel hit state each frame
        self.moon_panel_state = None

        self.draw_system_view()
        if (
            self.scope_mode == "local"
            and self.focus_body
            and self.focus_body.children
        ):
            self.draw_moon_panel(self.focus_body)
        self.draw_info_panel()
        self.draw_help_overlay()
        pygame.display.flip()

    def draw_system_view(self):
        scale = self.base_pixels_per_au * self.zoom

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
            col = (90, 90, 90) if b.is_belt() else (120, 100, 50)
            self.draw_orbit_for_body(b, scale, col)

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
                size = max(size, 4)
            if b.type in ("star", "primary_star"):
                size = max(size, 10)
            if b.type == "barycenter":
                size = 4
            pygame.draw.circle(self.screen, b.color, (int(sx), int(sy)), size)
            if b is self.focus_body:
                pygame.draw.circle(
                    self.screen, (255, 255, 255), (int(sx), int(sy)), size + 4, 1
                )

        # Measurement line
        if (
            self.measure_mode
            and self.measure_start_world is not None
            and self.measure_end_world is not None
        ):
            sx1, sy1 = self.world_to_screen(
                self.measure_start_world[0], self.measure_start_world[1], scale
            )
            sx2, sy2 = self.world_to_screen(
                self.measure_end_world[0], self.measure_end_world[1], scale
            )
            pygame.draw.line(
                self.screen, (200, 220, 255),
                (int(sx1), int(sy1)), (int(sx2), int(sy2)), 2
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
        panel_width = self.width - sidebar_width - 20
        panel_height = int(self.height * 0.30)
        rect = pygame.Rect(
            10,
            self.height - panel_height - 10,
            panel_width,
            panel_height,
        )
        pygame.draw.rect(self.screen, (8, 8, 20), rect)
        pygame.draw.rect(self.screen, (80, 80, 110), rect, 1)

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
        txt = self.small_font.render(label, True, (200, 200, 220))
        self.screen.blit(txt, (rect.x + 6, rect.y + 6))

        # store hit-test info for clicks
        self.moon_panel_state = {
            "rect": rect,
            "center_body": center_body,
            "moons": moon_hits,
        }

    def draw_info_panel(self):
        sidebar_width = int(self.width * 0.32)
        rect = pygame.Rect(self.width - sidebar_width, 0, sidebar_width, self.height)
        pygame.draw.rect(self.screen, (5, 5, 15), rect)
        pygame.draw.rect(self.screen, (60, 60, 90), rect, 1)

        lines: List[str] = []

        sys_id = getattr(self.system, "id", "?")
        sys_name = getattr(self.system, "name", sys_id)
        if self.system_defs:
            idx = 0
            for i, s in enumerate(self.system_defs):
                if s["id"] == self.current_system_id:
                    idx = i
                    break
            lines.append(f"System [{idx+1}/{len(self.system_defs)}]: {sys_name} ({sys_id})")
        else:
            lines.append(f"System: {sys_name} ({sys_id})")

        lines.append(f"View: {self.view_mode:<3}   Scope: {self.scope_mode:<6}")
        lines.append(
            f"Speed x{self.speed_multiplier:.3f}   Follow: {'on' if self.follow_focus else 'off'}"
        )

        b = self.focus_body
        if b and b.period_years > 0:
            seconds_per_orbit = BASE_ORBIT_SECONDS / self.speed_multiplier
            lines.append(f"~{seconds_per_orbit:.1f}s per orbit at this speed")

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
        lines.append(f"Measure mode: {'on' if self.measure_mode else 'off'}")
        if (
            self.measure_mode
            and self.measure_start_world is not None
            and self.measure_end_world is not None
        ):
            sx, sy = self.measure_start_world
            ex, ey = self.measure_end_world
            dx = ex - sx
            dy = ey - sy
            dist_au = math.hypot(dx, dy)
            if dist_au > 1e-6:
                dist_km = dist_au * AU_IN_KM
                lines.append(f"d ≈ {dist_au:.3f} AU  (~{dist_km:,.0f} km)")
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

        x = rect.x + 10
        y = rect.y + 10
        for line in lines[:26]:
            txt = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(txt, (x, y))
            y += txt.get_height() + 2

    def draw_help_overlay(self):
        help_lines = [
            "SPACE: play/pause   [ / ]: slower/faster   +/- or wheel: zoom   0: reset zoom+pan",
            "TAB/Shift+TAB/↑↓: focus   LEFT-DRAG: pan (normal) / tap: select   1/2/3: belts/moons/dwarfs   V: top/iso",
            "M: moon panel   F: follow focus   D: measure mode (L-drag)   , / .: prev/next system (universe.json)",
        ]
        y = self.height - 50
        for line in help_lines:
            txt = self.small_font.render(line, True, (150, 150, 150))
            self.screen.blit(txt, (10, y))
            y += txt.get_height() + 1


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
    system_defs = list_systems_in_universe(args.json_path)

    # Resolve current_system_id
    current_system_id = args.system_id
    if current_system_id is None:
        if system_defs:
            current_system_id = system_defs[0]["id"]
        else:
            current_system_id = getattr(system, "id", None)

    sandbox = OrbitSandbox(
        system,
        json_path=args.json_path,
        system_defs=system_defs,
        current_system_id=current_system_id,
    )
    sandbox.system.update(0.0)
    sandbox.run()


if __name__ == "__main__":
    main()
