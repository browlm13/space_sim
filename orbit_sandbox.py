#!/usr/bin/env python3
"""
orbit_sandbox.py

Minimal sandbox orbit viewer for fictional systems.

Features
--------
- Load either:
    * a system JSON: { "id": "...", "name": "...", "bodies": [...] }
    * a universe JSON: { "systems": [ { "id": "...", "bodies_file": "...", ... } ] }
- Animate circular orbits using Kepler-style T ~ a^(3/2).
- Hierarchical orbits: barycenter -> stars -> planets -> moons.
- Top-down AND isometric view modes (toggle with V).
- Always-visible system view.
- Optional moon-system mini panel for focused body with moons (M).
- Time scale automatically tied to the focused body's orbital period
  (by default 1 orbit ≈ BASE_ORBIT_SECONDS at speed x1).
- Time controls, zoom, focus, click-to-focus (bodies + orbits).
- Mouse-based measurement mode:
    - Press D to toggle measure mode.
    - In measure mode, first click selects origin body, second and later clicks select target.
    - Draws a line between origin and target and shows distance + travel times in sidebar.
- Right-hand wiki sidebar with orbit stats + gravity, population, atmosphere, species, etc.
  (moons get full wiki too, since they are just bodies with their own meta).

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
        self.a_au: float = float(raw.get("a", 0.0))  # semi-major axis in AU
        self.e: float = float(raw.get("e", 0.0))
        self.inclination_deg: float = float(raw.get("inclination", 0.0))

        # Physical/visual
        self.radius_km: float = float(raw.get("radius", 0.0))
        self.visual_size: int = int(raw.get("visual_size", 4))
        self.color: Tuple[int, int, int] = hex_to_rgb(raw.get("color", "#ffffff"))

        # Orbital period in years (if not provided, use Kepler-ish scaling)
        period = raw.get("period_years")
        if period is None and self.a_au > 0:
            self.period_years: float = self.a_au ** 1.5
        else:
            self.period_years = float(period or 0.0)

        # Initial phase in radians
        phase_deg = raw.get("phase_deg")
        if phase_deg is None:
            self.initial_phase: float = random.random() * 2.0 * math.pi
        else:
            self.initial_phase = math.radians(float(phase_deg))

        self.angle: float = self.initial_phase  # current mean anomaly / phase
        self.mean_motion: float = (
            2.0 * math.pi / self.period_years if self.period_years > 0 else 0.0
        )

        # Meta / lore
        self.tags: List[str] = raw.get("tags", [])
        self.image: Optional[str] = raw.get("image")  # optional art path
        self.meta: dict = raw.get("meta", {})

        # world-space position in AU
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

        # largest orbital radius for scaling
        max_a = 0.0
        for b in self.bodies.values():
            if not b.is_root():
                max_a = max(max_a, b.a_au)
        self.max_a_au = max_a if max_a > 0 else 1.0

    def update(self, dt_years: float):
        # Update angles
        for b in self.bodies.values():
            b.update_angle(dt_years)
        # Refresh world positions, respecting hierarchy
        for root in self.roots:
            self._update_positions_recursive(root)

    def _update_positions_recursive(self, body: Body):
        if body.is_root():
            body.pos = (0.0, 0.0)
        else:
            px, py = body.parent.pos
            r = body.a_au
            body.pos = (
                px + r * math.cos(body.angle),
                py + r * math.sin(body.angle),
            )
        for child in body.children:
            self._update_positions_recursive(child)

    def ordered_focusable_bodies(self) -> List[Body]:
        """Bodies you can cycle focus through: stars, planets, giants, dwarfs, etc."""
        focusables = [
            b
            for b in self.bodies.values()
            if not b.is_belt() and b.type != "barycenter"
        ]
        focusables.sort(key=lambda b: (b.type not in ("star", "primary_star"), b.a_au))
        return focusables


# ---------- Loading JSON ----------


def load_system_from_json(path: str, system_id: Optional[str] = None) -> SystemModel:
    """
    Load either:
      - a system JSON: { "id": "...", "name": "...", "bodies": [...] }
      - a universe JSON: { "systems": [ { "id": "...", "bodies_file": "...", ... } ] }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Universe file?
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

        # Bodies inline or via file
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

        # time scaling:
        #   base_time_scale = years per second for speed x1 (depends on focused body)
        #   speed_multiplier = global factor ([ / ])
        #   time_scale = base_time_scale * speed_multiplier
        self.base_time_scale = 1.0
        self.speed_multiplier = 1.0
        self.time_scale = 1.0

        # Projection: "top" or "iso"
        self.view_mode = "top"
        # Scope: "system" or "local" (local = show moon mini panel)
        self.scope_mode = "system"

        # scaling: base pixels per AU (before zoom)
        self.base_pixels_per_au = 0.44 * min(self.width, self.height) / system.max_a_au
        self.zoom = 1.0

        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)

        self.show_belts = True
        self.show_moons = True      # show moons by default
        self.show_dwarfs = True

        self.focusables = system.ordered_focusable_bodies()
        self.focus_index = 0 if self.focusables else -1
        self.focus_body: Optional[Body] = (
            self.focusables[self.focus_index] if self.focus_index >= 0 else None
        )

        # measurement state
        self.measure_active = False
        self.measure_origin: Optional[Body] = None
        self.measure_target: Optional[Body] = None

        # initialize time scale based on initial focus
        self.recalc_time_scale()

    # --- time scaling helper ---

    def recalc_time_scale(self):
        """Recompute time_scale from focused body's period and speed_multiplier."""
        b = self.focus_body
        if b and b.period_years > 0:
            # 1x speed: one orbit in BASE_ORBIT_SECONDS real seconds
            self.base_time_scale = b.period_years / BASE_ORBIT_SECONDS
        else:
            self.base_time_scale = 1.0
        self.time_scale = self.base_time_scale * self.speed_multiplier

    # --- main loop ---

    def run(self):
        while True:
            dt_real = self.clock.tick(60) / 1000.0  # seconds
            self.handle_events()
            if self.running:
                dt_years = dt_real * self.time_scale
                self.sim_time_years += dt_years
                self.system.update(dt_years)
            self.draw()

    # --- coordinate projection helpers ---

    def project_world(self, wx: float, wy: float, cam_x: float, cam_y: float, scale: float):
        """
        Convert world coords (AU) to screen coords (pixels)
        according to the current view_mode.
        """
        dx = wx - cam_x
        dy = wy - cam_y

        if self.view_mode == "top":
            sx = self.width / 2 + dx * scale
            sy = self.height / 2 + dy * scale
        else:  # isometric / tilted
            iso_x = (dx - dy) * scale * 0.75
            iso_y = (dx + dy) * scale * 0.40
            sx = self.width / 2 + iso_x
            sy = self.height / 2 + iso_y

        return sx, sy

    def draw_orbit_for_body(self, b: Body, cam_x: float, cam_y: float, scale: float, color):
        parent = b.parent
        if not parent:
            return

        if self.view_mode == "top":
            cx, cy = self.project_world(parent.pos[0], parent.pos[1], cam_x, cam_y, scale)
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
                sx, sy = self.project_world(wx, wy, cam_x, cam_y, scale)
                pts.append((int(sx), int(sy)))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, color, False, pts, 1)

    # --- event handling ---

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

            if event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_click(event.pos)

            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.zoom *= 1.1
                elif event.y < 0:
                    self.zoom /= 1.1
                self.zoom = max(0.1, min(self.zoom, 10.0))

    def handle_keydown(self, key):
        if key == pygame.K_SPACE:
            self.running = not self.running

        # Arrow keys: move focus inward/outward through sorted bodies
        if key == pygame.K_UP:
            self.cycle_focus(1)
        if key == pygame.K_DOWN:
            self.cycle_focus(-1)

        # Zoom
        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.zoom *= 1.1
        if key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.zoom /= 1.1
        self.zoom = max(0.1, min(self.zoom, 10.0))

        # Time scale multiplier
        if key == pygame.K_RIGHTBRACKET:  # ]
            self.speed_multiplier *= 2.0
            self.speed_multiplier = min(self.speed_multiplier, 128.0)
            self.recalc_time_scale()
        if key == pygame.K_LEFTBRACKET:  # [
            self.speed_multiplier /= 2.0
            self.speed_multiplier = max(self.speed_multiplier, 1.0 / 128.0)
            self.recalc_time_scale()

        # Reset zoom
        if key == pygame.K_0:
            self.zoom = 1.0

        # Focus cycling (TAB / SHIFT+TAB)
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

        # View mode toggle
        if key == pygame.K_v:
            self.view_mode = "iso" if self.view_mode == "top" else "top"

        # Scope toggle: moon mini panel on/off
        if key == pygame.K_m:
            if self.scope_mode == "system":
                if self.focus_body and self.focus_body.children:
                    self.scope_mode = "local"
            else:
                self.scope_mode = "system"

        # Measurement mode toggle
        if key == pygame.K_d:
            self.measure_active = not self.measure_active
            if not self.measure_active:
                self.measure_origin = None
                self.measure_target = None

    def handle_click(self, pos):
        """Click selection with:
        - big radius around body
        - orbit rings clickable too
        - in measure mode: first click = origin, second+ = target
        """
        mx, my = pos
        cam_x, cam_y = self.camera_center()
        scale = self.base_pixels_per_au * self.zoom

        best_dot_body = None
        best_dot_score = float("inf")
        best_orbit_body = None
        best_orbit_score = float("inf")

        for b in self.system.bodies.values():
            if b.is_belt():
                continue

            # Body hit area (bigger than the dot)
            sx, sy = self.project_world(b.pos[0], b.pos[1], cam_x, cam_y, scale)
            dx = sx - mx
            dy = sy - my
            d2 = dx * dx + dy * dy
            body_r = max(14, b.visual_size + 10)  # fat target
            if d2 <= body_r * body_r and d2 < best_dot_score:
                best_dot_score = d2
                best_dot_body = b

            # Orbit ring hit area
            if b.parent and b.a_au > 0:
                cx, cy = self.project_world(
                    b.parent.pos[0], b.parent.pos[1], cam_x, cam_y, scale
                )
                odx = mx - cx
                ody = my - cy
                dist_center = math.hypot(odx, ody)
                orbit_radius = b.a_au * scale
                if orbit_radius > 4:
                    diff = abs(dist_center - orbit_radius)
                    if diff < 10 and diff < best_orbit_score:
                        best_orbit_score = diff
                        best_orbit_body = b

        clicked_body = best_dot_body or best_orbit_body
        if clicked_body is None:
            return

        # Focus always follows click
        self.set_focus(clicked_body)

        # Measurement selection (if active)
        if self.measure_active:
            if self.measure_origin is None:
                self.measure_origin = clicked_body
                self.measure_target = None
            else:
                # if you click the same body, treat as new origin
                if clicked_body is self.measure_origin:
                    self.measure_target = None
                else:
                    self.measure_target = clicked_body

    def cycle_focus(self, direction: int):
        if not self.focusables:
            return
        self.focus_index = (self.focus_index + direction) % len(self.focusables)
        self.focus_body = self.focusables[self.focus_index]
        self.recalc_time_scale()

    def set_focus(self, body: Body):
        self.focus_body = body
        if body in self.focusables:
            self.focus_index = self.focusables.index(body)
        self.recalc_time_scale()

    def camera_center(self) -> Tuple[float, float]:
        if self.focus_body is None:
            return (0.0, 0.0)
        return self.focus_body.pos

    # --- drawing ---

    def draw(self):
        self.screen.fill((0, 0, 0))

        # Always show full system view
        self.draw_system_view()

        # Optional moon mini panel for focused body with moons
        if (
            self.scope_mode == "local"
            and self.focus_body
            and self.focus_body.children
        ):
            self.draw_moon_panel(self.focus_body)

        # Sidebar + help text
        self.draw_info_panel()
        self.draw_help_overlay()

        pygame.display.flip()

    def draw_system_view(self):
        cam_x, cam_y = self.camera_center()
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

            if b.is_belt():
                col = (90, 90, 90)
            else:
                col = (120, 100, 50)

            self.draw_orbit_for_body(b, cam_x, cam_y, scale, col)

        # Bodies
        for b in self.system.bodies.values():
            if b.is_belt():
                continue
            if b.is_moon() and not self.show_moons:
                continue
            if b.type == "dwarf_planet" and not self.show_dwarfs:
                continue

            sx, sy = self.project_world(b.pos[0], b.pos[1], cam_x, cam_y, scale)

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

        # Measurement line (draw after bodies so it's on top)
        if self.measure_active and self.measure_origin and self.measure_target:
            o = self.measure_origin
            t = self.measure_target
            ox, oy = self.project_world(o.pos[0], o.pos[1], cam_x, cam_y, scale)
            tx, ty = self.project_world(t.pos[0], t.pos[1], cam_x, cam_y, scale)
            pygame.draw.line(
                self.screen,
                (120, 220, 255),
                (int(ox), int(oy)),
                (int(tx), int(ty)),
                2,
            )

    def draw_moon_panel(self, center_body: Body):
        """Mini-sim showing the focused body's moons in their own frame."""
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
        for m in moons:
            r = int(m.a_au * local_scale)
            if r <= 1:
                continue
            pygame.draw.circle(self.screen, orbit_color, (cx, cy), r, 1)

        pygame.draw.circle(self.screen, center_body.color, (cx, cy), 6)

        for m in moons:
            x = cx + math.cos(m.angle) * m.a_au * local_scale
            y = cy + math.sin(m.angle) * m.a_au * local_scale
            pygame.draw.circle(self.screen, m.color, (int(x), int(y)), 3)

        label = f"{center_body.name} local system"
        txt = self.small_font.render(label, True, (200, 200, 220))
        self.screen.blit(txt, (rect.x + 6, rect.y + 6))

    def draw_info_panel(self):
        sidebar_width = int(self.width * 0.32)
        rect = pygame.Rect(self.width - sidebar_width, 0, sidebar_width, self.height)

        pygame.draw.rect(self.screen, (5, 5, 15), rect)
        pygame.draw.rect(self.screen, (60, 60, 90), rect, 1)

        lines: List[str] = []

        lines.append(f"System: {self.system.name}")
        measure_flag = "on" if self.measure_active else "off"
        lines.append(f"View: {self.view_mode:<3}   Scope: {self.scope_mode:<6}   Measure: {measure_flag}")
        lines.append(f"Speed x{self.speed_multiplier:.3f}")

        b = self.focus_body
        if b and b.period_years > 0:
            seconds_per_orbit = BASE_ORBIT_SECONDS / self.speed_multiplier
            lines.append(f"~{seconds_per_orbit:.1f}s per orbit at this speed")

        lines.append(f"Sim time: {self.sim_time_years:8.3f} years")

        if b:
            lines.append(f"Focus: {b.name}  [{b.type}]")

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

                for frac in (0.1, 0.01):
                    v = SPEED_OF_LIGHT_KM_S * frac
                    t_sec = a_km / v
                    t_days = t_sec / 86400.0
                    lines.append(f"Straight-line @ {frac:.0%} c: {t_days:.2f} days")
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

            factions = meta.get("factions")
            if isinstance(factions, list):
                factions_str = ", ".join(factions)
            else:
                factions_str = factions
            if factions_str:
                lines.append(f"factions: {factions_str}")

            img_path = b.image or meta.get("image")
            if img_path:
                lines.append(f"image: {img_path}")

            desc = meta.get("short_description")
            if desc:
                lines.append("")
                lines.append(desc)

        # Measurement block (origin → target)
        if (
            self.measure_active
            and self.measure_origin is not None
            and self.measure_target is not None
            and self.measure_origin is not self.measure_target
        ):
            o = self.measure_origin
            t = self.measure_target
            dx = (t.pos[0] - o.pos[0])
            dy = (t.pos[1] - o.pos[1])
            dist_au = math.hypot(dx, dy)
            dist_km = dist_au * AU_IN_KM

            lines.append("")
            lines.append(f"Measure: {o.name} → {t.name}")
            lines.append(f"d ≈ {dist_au:.3f} AU (~{dist_km:,.0f} km)")

            light_seconds = dist_km / SPEED_OF_LIGHT_KM_S if dist_km > 0 else 0.0
            if light_seconds > 0:
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
                if dist_km > 0:
                    t_sec = dist_km / v
                    t_days = t_sec / 86400.0
                    lines.append(f"@{frac:.0%} c: {t_days:.2f} d")

            if dist_km > 0:
                t_ship_sec = dist_km / SHIP_SPEED_KM_S
                t_ship_years = t_ship_sec / (86400.0 * 365.25)
                lines.append(f"@{SHIP_SPEED_KM_S:.0f} km/s: {t_ship_years:.2f} yr")

        x = rect.x + 10
        y = rect.y + 10
        for line in lines[:24]:
            txt = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(txt, (x, y))
            y += txt.get_height() + 2

    def draw_help_overlay(self):
        help_lines = [
            "SPACE: play/pause   [ / ]: slower/faster (relative to orbit)   +/- or wheel: zoom   0: reset zoom",
            "TAB / Shift+TAB / ↑↓: cycle focus   Click: body or orbit   M: toggle moon panel   1: belts  2: moons  3: dwarfs   V: top/iso   D: measure mode",
        ]
        y = self.height - 40
        for line in help_lines:
            txt = self.small_font.render(line, True, (150, 150, 150))
            self.screen.blit(txt, (10, y))
            y += txt.get_height() + 1


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser(description="Fictional solar system sandbox viewer")
    parser.add_argument(
        "json_path",
        help="Path to system JSON or universe JSON (see docstring for schema).",
    )
    parser.add_argument(
        "--system",
        dest="system_id",
        default=None,
        help="System id to load when using a universe.json file.",
    )
    args = parser.parse_args()

    system = load_system_from_json(args.json_path, system_id=args.system_id)
    sandbox = OrbitSandbox(system)
    sandbox.system.update(0.0)  # initialize positions
    sandbox.run()


if __name__ == "__main__":
    main()
