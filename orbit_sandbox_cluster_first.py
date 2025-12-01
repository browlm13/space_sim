#!/usr/bin/env python3
"""
orbit_sandbox_cluster_first.py

Enhanced space explorer with cluster-first interface.

IMPROVEMENTS:
- Cluster view is the startup/home screen (not system view)
- Better star imagery: gradient glows, spectral class colors, size by mass
- Galactic context: coordinates, orbital parameters, light-distances
- Integrated wiki panel accessible from cluster view
- Body window shows cluster selection UI
- Spatial orientation with proper light-year distances
- Relative motion indicators

Toggles
-------
G: toggle between CLUSTER view and SYSTEM view
, / . : previous / next system in the universe.json list
W: toggle wiki sidebar (cluster view)
D: measure mode with detailed distance information

Usage
-----
    python orbit_sandbox_cluster_first.py data/universe.json

Dependencies
-----------
    pip install pygame
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import pygame

AU_IN_KM = 149_597_870.7
LY_IN_KM = AU_IN_KM * 63241.077  # ~1 light-year in km
SHIP_SPEED_KM_S = 20.0
SPEED_OF_LIGHT_KM_S = 299_792.458
BASE_ORBIT_SECONDS = 20.0


# ---------- Star rendering helpers ----------

def get_spectral_color(spectral_class: str) -> Tuple[int, int, int]:
    """Map spectral class to realistic star color."""
    sc = spectral_class.upper() if spectral_class else "G2V"
    
    # Simplified spectral colors
    if sc.startswith("O"):
        return (155, 176, 255)  # Blue
    elif sc.startswith("B"):
        return (170, 191, 255)  # Blue-white
    elif sc.startswith("A"):
        return (202, 215, 255)  # White
    elif sc.startswith("F"):
        return (248, 247, 255)  # Yellow-white
    elif sc.startswith("G"):
        return (255, 244, 234)  # Yellow (Sun-like)
    elif sc.startswith("K"):
        return (255, 205, 110)  # Orange
    elif sc.startswith("M"):
        return (255, 116, 77)   # Red
    else:
        return (255, 255, 200)  # Default yellow


def get_star_size_visual(mass_solar: float) -> int:
    """Estimate visual size from stellar mass."""
    if mass_solar < 0.5:
        return 6
    elif mass_solar < 1.0:
        return 8
    elif mass_solar < 1.5:
        return 10
    elif mass_solar < 3.0:
        return 12
    else:
        return 14


def draw_star_glow(surface: pygame.Surface, pos: Tuple[int, int], radius: int, 
                   color: Tuple[int, int, int], intensity: float = 1.0):
    """Draw a star with realistic glow effect."""
    sx, sy = int(pos[0]), int(pos[1])
    
    # Multiple concentric circles for glow
    glow_colors = [
        tuple(int(c * 0.3 * intensity) for c in color),
        tuple(int(c * 0.6 * intensity) for c in color),
        color
    ]
    glow_radii = [radius + 6, radius + 3, radius]
    
    for glow_r, glow_c in zip(glow_radii, glow_colors):
        pygame.draw.circle(surface, glow_c, (sx, sy), glow_r)


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
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except ValueError:
        return (255, 255, 255)


class UIButton:
    """Simple rectangular UI button with hover and click handling."""
    def __init__(self, label: str, x: int, y: int, w: int, h: int, font: pygame.font.Font, callback):
        self.label = label
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.callback = callback
        self.hover = False

    def draw(self, surface: pygame.Surface):
        bg = (40, 44, 60) if not self.hover else (70, 80, 120)
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, (110, 120, 160), self.rect, 1, border_radius=6)
        txt = self.font.render(self.label, True, (220, 220, 240))
        tx = self.rect.x + (self.rect.w - txt.get_width()) // 2
        ty = self.rect.y + (self.rect.h - txt.get_height()) // 2
        surface.blit(txt, (tx, ty))

    def contains(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)

    def click(self):
        if callable(self.callback):
            self.callback()


class Body:
    """Celestial body with orbital and physical parameters."""
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
            import random
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
    """Solar system with hierarchical bodies."""
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


def load_system_from_json(path: str, system_id: Optional[str] = None) -> SystemModel:
    """Load a single system from universe.json or system.json."""
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


def load_universe_data(path: str) -> dict:
    """Load universe.json with all metadata."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"meta": {}, "systems": []}


def list_systems_in_universe(path: str) -> List[dict]:
    """Return list of system descriptors from universe.json."""
    data = load_universe_data(path)
    systems = []
    raw_systems = data.get("systems", [])

    for s in raw_systems:
        sid = s.get("id")
        if not sid:
            continue
        name = s.get("name", sid)
        offset = s.get("offset") or [0.0, 0.0]
        try:
            ox = float(offset[0])
            oy = float(offset[1])
        except Exception:
            ox, oy = 0.0, 0.0
        
        color_hex = s.get("color", "#ffffff")
        color = hex_to_rgb(color_hex)
        
        # Extract galactic metadata
        spectral = s.get("spectral_class", "G2V")
        masses = s.get("masses_solar", [1.0])
        main_mass = masses[0] if masses else 1.0
        
        systems.append({
            "id": sid,
            "name": name,
            "offset": (ox, oy),
            "color": color,
            "spectral_class": spectral,
            "mass": main_mass,
            "age_gy": s.get("age_gy", 4.5),
            "galactic_longitude": s.get("galactic_longitude", 0.0),
            "galactic_latitude": s.get("galactic_latitude", 0.0),
            "radial_velocity_kms": s.get("radial_velocity_kms", 0.0),
            "description": s.get("description", ""),
        })
    return systems


class ClusterViewer:
    """Cluster-first space exploration UI."""
    
    def __init__(self, universe_json_path: str, width: int = 1400, height: int = 900):
        pygame.init()
        pygame.display.set_caption("Cosmic Wrath – Star Cluster Explorer")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Load universe data
        self.json_path = universe_json_path
        self.universe_data = load_universe_data(universe_json_path)
        self.system_defs = list_systems_in_universe(universe_json_path)
        
        # Current system
        self.current_system_id = self.system_defs[0]["id"] if self.system_defs else "unknown"
        self.system: Optional[SystemModel] = None
        self.load_current_system()

        # Fonts
        self.font = pygame.font.SysFont("consolas", 14)
        self.small_font = pygame.font.SysFont("consolas", 12)
        self.title_font = pygame.font.SysFont("consolas", 24, bold=True)
        self.header_font = pygame.font.SysFont("consolas", 18, bold=True)
        self.tiny_font = pygame.font.SysFont("consolas", 10)

        # Camera (cluster view)
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.cluster_zoom = 1.0

        # Cluster extent calculation
        if self.system_defs:
            xs = [sd["offset"][0] for sd in self.system_defs]
            ys = [sd["offset"][1] for sd in self.system_defs]
            max_extent = max(
                max(abs(x) for x in xs) if xs else 1.0,
                max(abs(y) for y in ys) if ys else 1.0,
                2.0,
            )
        else:
            max_extent = 2.0
        self.cluster_pixels_per_ly = 0.35 * min(self.width * 0.65, self.height) / max_extent

        # Mode
        self.map_mode = "cluster"  # cluster or system
        self.show_wiki_sidebar = True

        # System view camera
        self.system_cam_x = 0.0
        self.system_cam_y = 0.0
        self.system_zoom = 1.0
        self.system_view_mode = "top"
        
        # Simulation
        self.system_sim_time = 0.0
        self.system_running = False
        self.system_time_scale = 1.0
        self.system_speed_mult = 1.0
        
        # System view focus
        self.focus_body: Optional[Body] = None
        self.focusables: List[Body] = []
        self.focus_index = -1
        
        # Interaction
        self.left_dragging = False
        self.drag_start_screen = (0, 0)
        self.drag_start_cam = (0.0, 0.0)
        
        # Measurement
        self.measure_mode = False
        self.measure_dragging = False
        self.measure_start = (0.0, 0.0)
        self.measure_end = (0.0, 0.0)
        
        # Selection in cluster view
        self.selected_system_idx = 0
        self.hovered_idx = None

        # UI buttons (will be positioned relative to map area)
        self.ui_buttons: List[UIButton] = []
        self._init_ui_buttons()
        # Modal state for listing bodies in a system
        self.show_bodies_modal = False
        self._current_modal_bodies: List[Body] = []
        
        self.running = True

    def load_current_system(self):
        """Load the currently selected system."""
        if self.system:
            return  # Only load once for cluster view
        self.system = load_system_from_json(self.json_path, system_id=self.current_system_id)
        self.system.update(0.0)
        self.focusables = self.system.ordered_focusable_bodies()
        if self.focusables:
            self.focus_body = self.focusables[0]
            self.focus_index = 0
            self.system_cam_x, self.system_cam_y = self.focus_body.pos
        else:
            self.focus_body = None
            self.focus_index = -1

    def _init_ui_buttons(self):
        # Place buttons at top-right of the cluster map area
        map_width = int(self.width * 0.65)
        bx = int(map_width - 160)
        by = 16
        bw = 160
        bh = 36
        f = self.small_font

        self.ui_buttons = [
            UIButton("Explore", bx, by, bw, bh, f, self._btn_dive),
            UIButton("Bodies", bx, by + 46, bw, bh, f, self._btn_bodies),
            UIButton("Wiki", bx, by + 92, bw, bh, f, self._btn_wiki),
            UIButton("Measure", bx, by + 138, bw, bh, f, self._btn_measure),
            UIButton("Reset", bx, by + 184, bw, bh, f, self._btn_reset),
            UIButton("Quit", bx, by + 230, bw, bh, f, self._btn_quit),
        ]

    # UI button callbacks
    def _btn_dive(self):
        # Dive into hovered or selected system
        idx = self.hovered_idx if self.hovered_idx is not None else self.selected_system_idx
        if idx is not None and 0 <= idx < len(self.system_defs):
            sd = self.system_defs[idx]
            self.current_system_id = sd["id"]
            self.map_mode = "system"

    def _btn_bodies(self):
        # Toggle bodies modal for the hovered/selected system
        idx = self.hovered_idx if self.hovered_idx is not None else self.selected_system_idx
        if idx is None or not (0 <= idx < len(self.system_defs)):
            return
        sd = self.system_defs[idx]
        # Load system bodies for the modal
        try:
            system = load_system_from_json(self.json_path, system_id=sd["id"])
            self._current_modal_bodies = list(system.ordered_focusable_bodies())
        except Exception:
            self._current_modal_bodies = []
        self.show_bodies_modal = not getattr(self, "show_bodies_modal", False)

    def _btn_wiki(self):
        self.show_wiki_sidebar = not self.show_wiki_sidebar

    def _btn_measure(self):
        self.measure_mode = not self.measure_mode
        if not self.measure_mode:
            self.measure_dragging = False

    def _btn_reset(self):
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.cluster_zoom = 1.0

    def _btn_quit(self):
        self.running = False

    def world_to_screen_cluster(self, wx: float, wy: float) -> Tuple[float, float]:
        """Transform cluster world coords to screen coords."""
        dx = wx - self.cam_x
        dy = wy - self.cam_y
        sx = self.width * 0.65 / 2 + dx * self.cluster_pixels_per_ly * self.cluster_zoom
        sy = self.height / 2 + dy * self.cluster_pixels_per_ly * self.cluster_zoom
        return sx, sy

    def screen_to_world_cluster(self, sx: float, sy: float) -> Tuple[float, float]:
        """Transform screen coords to cluster world coords."""
        dx = (sx - self.width * 0.65 / 2) / (self.cluster_pixels_per_ly * self.cluster_zoom)
        dy = (sy - self.height / 2) / (self.cluster_pixels_per_ly * self.cluster_zoom)
        wx = self.cam_x + dx
        wy = self.cam_y + dy
        return wx, wy

    def draw_cluster_view(self):
        """Draw the cluster map view with stars and galactic context."""
        # Left panel (cluster map)
        map_width = int(self.width * 0.65)
        map_rect = pygame.Rect(0, 0, map_width, self.height)
        pygame.draw.rect(self.screen, (8, 8, 20), map_rect)
        pygame.draw.rect(self.screen, (60, 80, 120), map_rect, 2)

        # Title
        title = self.header_font.render("LOCAL STAR CLUSTER", True, (200, 200, 255))
        self.screen.blit(title, (20, 20))

        # Galactic context header
        meta = self.universe_data.get("meta", {})
        context = meta.get("galactic_context", {})
        region = context.get("region", "Unknown")
        dist_core = context.get("distance_from_core_kly", 0)
        dist_plane = context.get("distance_from_galactic_plane_ly", 0)
        
        context_text = f"Orion Spur · {dist_core:.1f} kly from galactic core · {dist_plane} ly from plane"
        ctx_surf = self.small_font.render(context_text, True, (180, 160, 200))
        self.screen.blit(ctx_surf, (20, 50))

        # Instructions
        instr = self.tiny_font.render("Click star to dive in  ·  Scroll to zoom  ·  Drag to pan  ·  M: measure mode", True, (150, 150, 180))
        self.screen.blit(instr, (20, 70))

        # Get mouse for hover
        mouse_pos = pygame.mouse.get_pos()
        hovered_idx = None
        
        # Draw grid/background reference
        grid_color = (20, 25, 40)
        grid_spacing = 50
        for x in range(0, map_width, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (0, y), (map_width, y))

        # Draw systems
        for idx, sd in enumerate(self.system_defs):
            ox, oy = sd["offset"]
            sx, sy = self.world_to_screen_cluster(ox, oy)
            
            # Skip if off screen
            if sx < 0 or sx > map_width or sy < 0 or sy > self.height:
                continue
            
            # Check hover
            dx = sx - mouse_pos[0]
            dy = sy - mouse_pos[1]
            d2 = dx * dx + dy * dy
            if d2 <= (25 * 25):
                hovered_idx = idx
            
            # Star glow and body
            spectral = sd.get("spectral_class", "G2V")
            star_color = get_spectral_color(spectral)
            mass = sd.get("mass", 1.0)
            star_radius = get_star_size_visual(mass)
            
            # Draw glow
            glow_intensity = 1.5 if idx == hovered_idx else 1.0
            if hovered_idx == idx:
                glow_intensity = 1.8
            
            # Multiple glow layers
            for glow_r, glow_alpha in [(15, 0.2), (10, 0.4), (5, 0.6)]:
                if hovered_idx == idx:
                    glow_alpha *= 1.3
                glow_color = tuple(int(c * glow_alpha) for c in star_color)
                pygame.draw.circle(self.screen, glow_color, (int(sx), int(sy)), int(glow_r))
            
            # Star body
            pygame.draw.circle(self.screen, star_color, (int(sx), int(sy)), star_radius)
            
            # Current system highlight
            if sd["id"] == self.current_system_id:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(sx), int(sy)), star_radius + 6, 2)
            
            # Hover highlight
            if hovered_idx == idx:
                pygame.draw.circle(self.screen, (255, 200, 100), (int(sx), int(sy)), star_radius + 8, 2)
            
            # System label
            label = sd["name"]
            label_surf = self.font.render(label, True, (210, 210, 230))
            self.screen.blit(label_surf, (int(sx) + star_radius + 12, int(sy) - 10))
            
            # Distance from current (if multiple systems)
            if len(self.system_defs) > 1:
                current_sd = self.system_defs[0]
                for i, check_sd in enumerate(self.system_defs):
                    if check_sd["id"] == self.current_system_id:
                        current_sd = check_sd
                        break
                cx, cy = current_sd["offset"]
                dist = math.hypot(ox - cx, oy - cy)
                dist_text = self.tiny_font.render(f"{dist:.2f} ly", True, (180, 180, 200))
                self.screen.blit(dist_text, (int(sx) - 20, int(sy) + star_radius + 12))
        # store hovered index for use by UI callbacks
        self.hovered_idx = hovered_idx

        # Draw UI buttons on top of map
        for btn in self.ui_buttons:
            # Update hover state based on mouse
            btn.hover = btn.contains(mouse_pos)
            btn.draw(self.screen)

        # If modal open, draw bodies list
        if getattr(self, "show_bodies_modal", False):
            self._draw_bodies_modal()

        # Measurement in cluster space
        if self.measure_mode and self.measure_dragging:
            ax, ay = self.measure_start
            bx, by = self.measure_end
            sx1, sy1 = self.world_to_screen_cluster(ax, ay)
            sx2, sy2 = self.world_to_screen_cluster(bx, by)
            pygame.draw.line(self.screen, (200, 220, 255), (int(sx1), int(sy1)), (int(sx2), int(sy2)), 2)
            
            # Distance info
            dist_ly = math.hypot(bx - ax, by - ay)
            if dist_ly > 0.01:
                label = f"{dist_ly:.3f} ly"
                label_surf = self.font.render(label, True, (200, 220, 255))
                mid_x = (sx1 + sx2) / 2
                mid_y = (sy1 + sy2) / 2
                self.screen.blit(label_surf, (int(mid_x) + 5, int(mid_y) - 10))

        # Right panel (wiki/info)
        if self.show_wiki_sidebar:
            wiki_x = int(self.width * 0.65)
            wiki_width = self.width - wiki_x
            wiki_rect = pygame.Rect(wiki_x, 0, wiki_width, self.height)
            pygame.draw.rect(self.screen, (5, 5, 15), wiki_rect)
            pygame.draw.rect(self.screen, (60, 60, 90), wiki_rect, 1)
            
            self.draw_wiki_panel(wiki_x, wiki_width)

    def draw_wiki_panel(self, x: int, width: int):
        """Draw the info/wiki sidebar."""
        pad = 10
        current_sd = None
        for sd in self.system_defs:
            if sd["id"] == self.current_system_id:
                current_sd = sd
                break
        
        if not current_sd:
            return
        
        y = 20
        
        # System name
        title = self.header_font.render(current_sd["name"], True, (200, 200, 255))
        self.screen.blit(title, (x + pad, y))
        y += title.get_height() + 10
        
        # Spectral class
        spec = current_sd.get("spectral_class", "Unknown")
        spec_surf = self.font.render(f"Spectral: {spec}", True, (210, 210, 230))
        self.screen.blit(spec_surf, (x + pad, y))
        y += spec_surf.get_height() + 5
        
        # Mass
        mass = current_sd.get("mass", 1.0)
        mass_surf = self.font.render(f"Mass: {mass:.2f} M☉", True, (210, 210, 230))
        self.screen.blit(mass_surf, (x + pad, y))
        y += mass_surf.get_height() + 5
        
        # Age
        age = current_sd.get("age_gy", 4.5)
        age_surf = self.font.render(f"Age: {age:.1f} Gy", True, (210, 210, 230))
        self.screen.blit(age_surf, (x + pad, y))
        y += age_surf.get_height() + 10
        
        # Galactic coordinates
        gal_lon = current_sd.get("galactic_longitude", 0)
        gal_lat = current_sd.get("galactic_latitude", 0)
        coord_text = f"Galactic: ({gal_lon:.1f}°, {gal_lat:.1f}°)"
        coord_surf = self.small_font.render(coord_text, True, (180, 180, 200))
        self.screen.blit(coord_surf, (x + pad, y))
        y += coord_surf.get_height() + 5
        
        # Radial velocity
        rv = current_sd.get("radial_velocity_kms", 0)
        rv_text = f"Radial velocity: {rv:+.1f} km/s"
        rv_surf = self.small_font.render(rv_text, True, (180, 180, 200))
        self.screen.blit(rv_surf, (x + pad, y))
        y += rv_surf.get_height() + 15
        
        # Description
        desc = current_sd.get("description", "No description available.")
        desc_lines = desc.split('\n')
        for line in desc_lines:
            if line.strip():
                desc_surf = self.font.render(line.strip(), True, (200, 200, 220))
                self.screen.blit(desc_surf, (x + pad, y))
                y += desc_surf.get_height() + 3
        
        y += 15
        
        # Suggested exploration
        explore_title = self.font.render("Stellar Features:", True, (150, 200, 255))
        self.screen.blit(explore_title, (x + pad, y))
        y += explore_title.get_height() + 8
        
        # Sample features based on spectral class
        features = {
            "O": ["Massive blue giant", "Extremely hot surface", "Short lifespan (few Myr)"],
            "B": ["Bright blue star", "Hot surface (10k K)", "Common in clusters"],
            "A": ["White star", "Hot (7.5k K)", "Sirius-like"],
            "F": ["Yellow-white star", "Medium-hot (6k K)", "Procyon-like"],
            "G": ["Sun-like star", "Moderate temp (5.8k K)", "Habitable zone likely"],
            "K": ["Orange dwarf", "Cool (3.9k K)", "Long-lived, many planets"],
            "M": ["Red dwarf", "Cool (3.1k K)", "Most common, long-lived"],
        }
        spec_key = spec[0] if spec else "G"
        feature_list = features.get(spec_key, ["Unknown stellar type"])
        
        for feature in feature_list:
            feature_surf = self.small_font.render(f"  • {feature}", True, (180, 180, 200))
            self.screen.blit(feature_surf, (x + pad, y))
            y += feature_surf.get_height() + 3

    def handle_events(self):
        """Handle user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_left_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.handle_left_up(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                if self.map_mode == "cluster":
                    if event.y > 0:
                        self.cluster_zoom *= 1.1
                    elif event.y < 0:
                        self.cluster_zoom /= 1.1
                    self.cluster_zoom = max(0.5, min(self.cluster_zoom, 5.0))

    def handle_keydown(self, key):
        """Handle keyboard input."""
        if key == pygame.K_ESCAPE:
            self.running = False
        
        if key == pygame.K_g:
            self.map_mode = "cluster" if self.map_mode == "system" else "system"
        
        if key == pygame.K_w:
            self.show_wiki_sidebar = not self.show_wiki_sidebar
        
        if key == pygame.K_d:
            self.measure_mode = not self.measure_mode
            if not self.measure_mode:
                self.measure_dragging = False
        
        if key == pygame.K_COMMA:
            self.switch_system(-1)
        if key == pygame.K_PERIOD:
            self.switch_system(1)

    def handle_left_down(self, pos):
        """Handle mouse down."""
        if self.measure_mode and self.map_mode == "cluster":
            self.measure_dragging = True
            wx, wy = self.screen_to_world_cluster(pos[0], pos[1])
            self.measure_start = (wx, wy)
            self.measure_end = (wx, wy)
        else:
            self.left_dragging = True
            self.drag_start_screen = pos
            if self.map_mode == "cluster":
                self.drag_start_cam = (self.cam_x, self.cam_y)

    def handle_left_up(self, pos):
        """Handle mouse up."""
        if self.measure_mode and self.measure_dragging:
            self.measure_dragging = False
            wx, wy = self.screen_to_world_cluster(pos[0], pos[1])
            self.measure_end = (wx, wy)
        elif self.left_dragging and self.map_mode == "cluster":
            dx = pos[0] - self.drag_start_screen[0]
            dy = pos[1] - self.drag_start_screen[1]
            d2 = dx * dx + dy * dy
            
            if d2 <= 100:  # Click threshold
                # First check UI buttons
                for btn in self.ui_buttons:
                    if btn.contains(pos):
                        btn.click()
                        self.left_dragging = False
                        return

                # If bodies modal is open, check modal clicks
                if getattr(self, "show_bodies_modal", False):
                    if self._handle_bodies_modal_click(pos):
                        self.left_dragging = False
                        return

                # Try to select a system
                wx, wy = self.screen_to_world_cluster(pos[0], pos[1])
                for idx, sd in enumerate(self.system_defs):
                    ox, oy = sd["offset"]
                    dist2 = (wx - ox) ** 2 + (wy - oy) ** 2
                    if dist2 <= (0.5 ** 2):  # 0.5 ly hit radius
                        self.current_system_id = sd["id"]
                        self.map_mode = "system"
                        break
            else:
                # Pan
                map_width = int(self.width * 0.65)
                px_per_ly = self.cluster_pixels_per_ly * self.cluster_zoom
                self.cam_x = self.drag_start_cam[0] - dx / px_per_ly
                self.cam_y = self.drag_start_cam[1] - dy / px_per_ly
            
            self.left_dragging = False

    def handle_mouse_motion(self, pos):
        """Handle mouse motion."""
        if self.left_dragging and not self.measure_mode and self.map_mode == "cluster":
            dx = pos[0] - self.drag_start_screen[0]
            dy = pos[1] - self.drag_start_screen[1]
            map_width = int(self.width * 0.65)
            px_per_ly = self.cluster_pixels_per_ly * self.cluster_zoom
            self.cam_x = self.drag_start_cam[0] - dx / px_per_ly
            self.cam_y = self.drag_start_cam[1] - dy / px_per_ly
        
        if self.measure_dragging and self.measure_mode:
            wx, wy = self.screen_to_world_cluster(pos[0], pos[1])
            self.measure_end = (wx, wy)

    def switch_system(self, delta: int):
        """Switch to next/previous system."""
        try:
            idx = [sd["id"] for sd in self.system_defs].index(self.current_system_id)
        except ValueError:
            idx = 0
        idx = (idx + delta) % len(self.system_defs)
        self.current_system_id = self.system_defs[idx]["id"]

    def draw(self):
        """Main draw function."""
        self.screen.fill((0, 0, 0))
        
        if self.map_mode == "cluster":
            self.draw_cluster_view()
        else:
            # System view placeholder
            # Basic system view implementation: orbits, bodies, and wiki/body windows
            self._draw_system_view()
        
        # Controls overlay
        help_lines = [
            "G: cluster/system view  W: toggle wiki  D: measure  ,/..: prev/next system  ESC: quit",
        ]
        y = self.height - 30
        for line in help_lines:
            txt = self.tiny_font.render(line, True, (150, 150, 150))
            self.screen.blit(txt, (10, y))
        
        pygame.display.flip()

    # ----------------- Bodies modal UI -----------------
    def _draw_bodies_modal(self):
        # center modal
        mw = int(self.width * 0.5)
        mh = int(self.height * 0.6)
        mx = (self.width - mw) // 2
        my = (self.height - mh) // 2
        rect = pygame.Rect(mx, my, mw, mh)
        pygame.draw.rect(self.screen, (18, 18, 30), rect)
        pygame.draw.rect(self.screen, (120, 120, 160), rect, 2)

        title = self.header_font.render("Bodies in system", True, (220, 220, 255))
        self.screen.blit(title, (mx + 16, my + 12))

        # list bodies
        y = my + 48
        item_h = 28
        for i, b in enumerate(self._current_modal_bodies[:18]):
            item_rect = pygame.Rect(mx + 16, y, mw - 32, item_h)
            pygame.draw.rect(self.screen, (30, 34, 50), item_rect)
            pygame.draw.rect(self.screen, (80, 80, 110), item_rect, 1)
            txt = self.font.render(f"{b.name}  [{b.type}]", True, (220, 220, 230))
            self.screen.blit(txt, (item_rect.x + 8, item_rect.y + 4))
            y += item_h + 6

    def _handle_bodies_modal_click(self, pos: Tuple[int, int]) -> bool:
        # Returns True if click was handled
        mw = int(self.width * 0.5)
        mh = int(self.height * 0.6)
        mx = (self.width - mw) // 2
        my = (self.height - mh) // 2
        y = my + 48
        item_h = 28
        for i, b in enumerate(self._current_modal_bodies[:18]):
            item_rect = pygame.Rect(mx + 16, y, mw - 32, item_h)
            if item_rect.collidepoint(pos):
                # open system view focused on this body
                self.current_system_id = b.system
                # load the system and set focus to this body
                try:
                    new_sys = load_system_from_json(self.json_path, system_id=self.current_system_id)
                    new_sys.update(0.0)
                    self.system = new_sys
                    # find the matching body object and set focus
                    focusables = self.system.ordered_focusable_bodies()
                    match = None
                    for idx_f, fb in enumerate(focusables):
                        if fb.id == b.id:
                            match = idx_f
                            break
                    if match is not None:
                        self.focus_body = focusables[match]
                    else:
                        self.focus_body = focusables[0] if focusables else None
                except Exception:
                    pass
                self.show_bodies_modal = False
                self.map_mode = "system"
                return True
            y += item_h + 6
        # Close modal if clicked outside
        modal_rect = pygame.Rect(mx, my, mw, mh)
        if not modal_rect.collidepoint(pos):
            self.show_bodies_modal = False
            return True
        return False

    # ----------------- Basic system view -----------------
    def _draw_system_view(self):
        # If system not loaded or doesn't match, load it
        try:
            if not self.system or getattr(self.system, "id", None) != self.current_system_id:
                self.system = load_system_from_json(self.json_path, system_id=self.current_system_id)
                self.system.update(0.0)
                self.focusables = self.system.ordered_focusable_bodies()
                self.focus_body = self.focusables[0] if self.focusables else None
        except Exception:
            txt = self.title_font.render("Failed to load system.", True, (200, 100, 100))
            self.screen.blit(txt, (self.width // 2 - txt.get_width() // 2, self.height // 2))
            return

        # simple orbit rendering in the left 2/3 of the screen
        sim_w = int(self.width * 0.68)
        sim_h = self.height
        sim_rect = pygame.Rect(0, 0, sim_w, sim_h)
        pygame.draw.rect(self.screen, (6, 6, 12), sim_rect)

        # Draw orbits
        scale = 0.6 * min(sim_w, sim_h) / max(1.0, self.system.max_a_au)
        cx = sim_rect.centerx
        cy = sim_rect.centery
        for b in self.system.bodies.values():
            if b.is_root():
                continue
            if b.is_belt():
                # draw belt band
                r = int(b.a_au * scale)
                color = (80, 80, 100)
                pygame.draw.circle(self.screen, color, (cx, cy), r, 2)
            else:
                r = int(b.a_au * scale)
                color = (80, 100, 140)
                pygame.draw.circle(self.screen, color, (cx, cy), r, 1)

        # Draw bodies
        for b in self.system.bodies.values():
            sx = cx + int(b.pos[0] * scale)
            sy = cy + int(b.pos[1] * scale)
            size = max(3, b.visual_size)
            pygame.draw.circle(self.screen, b.color, (sx, sy), size)
            if self.focus_body and b.id == getattr(self.focus_body, "id", None):
                pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), size + 4, 1)

        # Right sidebar: wiki/body info
        wiki_x = sim_w
        wiki_w = self.width - wiki_x
        wiki_rect = pygame.Rect(wiki_x, 0, wiki_w, self.height)
        pygame.draw.rect(self.screen, (5, 5, 15), wiki_rect)
        pygame.draw.rect(self.screen, (60, 60, 90), wiki_rect, 1)

        # Show focused body info
        y = 20
        if self.focus_body:
            title = self.header_font.render(self.focus_body.name, True, (220, 220, 255))
            self.screen.blit(title, (wiki_x + 12, y))
            y += title.get_height() + 8

            typ = self.font.render(f"Type: {self.focus_body.type}", True, (200, 200, 220))
            self.screen.blit(typ, (wiki_x + 12, y))
            y += typ.get_height() + 6

            a_text = f"a = {self.focus_body.a_au:.3f} AU" if self.focus_body.a_au > 0 else "Central body"
            a_surf = self.font.render(a_text, True, (200, 200, 220))
            self.screen.blit(a_surf, (wiki_x + 12, y))
            y += a_surf.get_height() + 6

            desc = self.focus_body.meta.get("short_description") if getattr(self.focus_body, "meta", None) else None
            if desc:
                lines = desc.split("\n")
                for line in lines:
                    txt = self.small_font.render(line, True, (200, 200, 220))
                    self.screen.blit(txt, (wiki_x + 12, y))
                    y += txt.get_height() + 4

        else:
            txt = self.font.render("No focus body", True, (180, 180, 180))
            self.screen.blit(txt, (wiki_x + 12, y))

    def run(self):
        """Main loop."""
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.draw()
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Cluster-first space explorer")
    parser.add_argument("json_path", help="Path to universe.json")
    args = parser.parse_args()
    
    viewer = ClusterViewer(args.json_path)
    viewer.run()


if __name__ == "__main__":
    main()
