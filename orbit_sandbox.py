#!/usr/bin/env python3
"""
orbit_sandbox.py

Minimal sandbox orbit viewer for fictional systems.

Features
--------
- Load either:
    * a system JSON: { "id": "...", "name": "...", "bodies": [...] }
    * a universe JSON: { "systems": [ { "id": "...", "bodies_file": "...", ... }, ... ] }
- Animate circular orbits using Kepler-style T ~ a^(3/2).
- Hierarchical orbits: barycenter -> stars -> planets -> moons.
- Top-down AND isometric view modes (toggle with V).
- Time controls, zoom, focus, click-to-focus.
- Info panel with orbit stats and simple lore metadata.

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
SHIP_SPEED_KM_S = 20.0  # for "time to go once around"


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
        self.image: Optional[str] = raw.get("image")
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
        self.time_scale = 1.0  # 1.0 = "1 sim year per real second" (roughly)

        # NEW: view mode – "top" or "iso"
        self.view_mode = "top"

        # scaling: base pixels per AU (before zoom)
        self.base_pixels_per_au = 0.44 * min(self.width, self.height) / system.max_a_au
        self.zoom = 1.0

        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)

        self.show_belts = True
        self.show_moons = False
        self.show_dwarfs = True

        self.focusables = system.ordered_focusable_bodies()
        self.focus_index = 0 if self.focusables else -1
        self.focus_body: Optional[Body] = (
            self.focusables[self.focus_index] if self.focus_index >= 0 else None
        )

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
            # simple fake-3D projection: rotate & squash
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
            # approximate orbit with a polyline ellipse
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

            # Mouse wheel zoom
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
            self.cycle_focus(1)      # outward
        if key == pygame.K_DOWN:
            self.cycle_focus(-1)     # inward


        # Zoom
        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.zoom *= 1.1
        if key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.zoom /= 1.1
        self.zoom = max(0.1, min(self.zoom, 10.0))

        # Time scale
        if key == pygame.K_RIGHTBRACKET:  # ]
            self.time_scale *= 2.0
        if key == pygame.K_LEFTBRACKET:  # [
            self.time_scale /= 2.0
        self.time_scale = max(0.001, min(self.time_scale, 1_000.0))

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

        # NEW: view mode toggle
        if key == pygame.K_v:
            self.view_mode = "iso" if self.view_mode == "top" else "top"

    def handle_click(self, pos):
        mx, my = pos
        cam_x, cam_y = self.camera_center()
        scale = self.base_pixels_per_au * self.zoom

        best_body = None
        best_dist2 = float("inf")
        for b in self.system.bodies.values():
            if b.is_belt():
                continue

            sx, sy = self.project_world(b.pos[0], b.pos[1], cam_x, cam_y, scale)
            dx = sx - mx
            dy = sy - my
            d2 = dx * dx + dy * dy
            r = max(5, b.visual_size + 4)
            if d2 <= r * r and d2 < best_dist2:
                best_dist2 = d2
                best_body = b
        if best_body:
            self.set_focus(best_body)

    def cycle_focus(self, direction: int):
        if not self.focusables:
            return
        self.focus_index = (self.focus_index + direction) % len(self.focusables)
        self.focus_body = self.focusables[self.focus_index]

    def set_focus(self, body: Body):
        self.focus_body = body
        if body in self.focusables:
            self.focus_index = self.focusables.index(body)

    def camera_center(self) -> Tuple[float, float]:
        if self.focus_body is None:
            return (0.0, 0.0)
        return self.focus_body.pos

    # --- drawing ---

    def draw(self):
        self.screen.fill((0, 0, 0))
        cam_x, cam_y = self.camera_center()
        scale = self.base_pixels_per_au * self.zoom

        # Draw orbits first
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

        # Draw bodies
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

            # focus halo
            if b is self.focus_body:
                pygame.draw.circle(
                    self.screen, (255, 255, 255), (int(sx), int(sy)), size + 4, 1
                )

        # Moon system mini-view
        if self.focus_body and self.focus_body.children:
            self.draw_moon_view(self.focus_body)

        # Info panel & help
        self.draw_info_panel()
        self.draw_help_overlay()

        pygame.display.flip()

    def draw_moon_view(self, planet: Body):
        rect = pygame.Rect(
            int(self.width * 0.60),
            int(self.height * 0.55),
            int(self.width * 0.38),
            int(self.height * 0.40),
        )
        pygame.draw.rect(self.screen, (10, 10, 10), rect)
        pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)

        moons = [m for m in planet.children if not m.is_belt()]
        if not moons:
            return

        max_a = max(m.a_au for m in moons) or 1.0
        local_scale = 0.4 * min(rect.width, rect.height) / max_a

        cx = rect.centerx
        cy = rect.centery

        # Orbits
        for m in moons:
            r = int(m.a_au * local_scale)
            if r <= 1:
                continue
            pygame.draw.circle(self.screen, (70, 70, 70), (cx, cy), r, 1)

        # Planet in center
        pygame.draw.circle(self.screen, planet.color, (cx, cy), 6)

        # Moons
        for m in moons:
            x = cx + math.cos(m.angle) * m.a_au * local_scale
            y = cy + math.sin(m.angle) * m.a_au * local_scale
            pygame.draw.circle(self.screen, m.color, (int(x), int(y)), 4)

        label = f"{planet.name} moon system"
        txt = self.small_font.render(label, True, (200, 200, 200))
        self.screen.blit(txt, (rect.x + 5, rect.y + 5))

    def draw_info_panel(self):
        lines: List[str] = []
        lines.append(f"System: {self.system.name}")
        lines.append(
            f"View: {self.view_mode:<3}   Sim time: {self.sim_time_years:8.3f} years   speed x{self.time_scale:.3f}"
        )
        if self.focus_body:
            b = self.focus_body
            lines.append(f"Focus: {b.name}  [{b.type}]")
            if b.a_au > 0:
                period = b.period_years
                a_km = b.a_au * AU_IN_KM
                circumference = 2.0 * math.pi * a_km
                ship_time_sec = circumference / SHIP_SPEED_KM_S
                ship_days = ship_time_sec / 86400.0
                lines.append(f"a = {b.a_au:.3f} AU   T = {period:.3f} years")
                lines.append(
                    f"Orbit length ~ {circumference:,.0f} km; "
                    f"{ship_days:,.1f} days @ {SHIP_SPEED_KM_S:.0f} km/s"
                )
            else:
                lines.append("a = 0 (central body)")

            g = b.meta.get("gravity_g")
            pop = b.meta.get("population_estimate")
            if g is not None or pop is not None:
                bits = []
                if g is not None:
                    bits.append(f"g ≈ {g}")
                if pop is not None:
                    bits.append(f"pop ~ {pop}")
                lines.append(" / ".join(bits))

            desc = b.meta.get("short_description")
            if desc:
                lines.append(desc)

        # Render
        x = 10
        y = 10
        for line in lines[:8]:
            txt = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(txt, (x, y))
            y += txt.get_height() + 2

    def draw_help_overlay(self):
        help_lines = [
            "SPACE: play/pause   [ / ]: slower / faster time   +/- or wheel: zoom   0: reset zoom",
            "TAB / Shift+TAB: cycle focus   Click body: focus   1: belts  2: moons  3: dwarfs   V: view top/iso",
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
