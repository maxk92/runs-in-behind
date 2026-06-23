"""
===============================================================
 Pitch Control Toolkit
 Based on: Fernandez & Bornn (2018) - Wide Open Spaces
===============================================================
Architecture:
    BaseDataProvider  (abstract)
        └── DFLDataProvider   (DFL Bundesliga open data XML)
        └── [YourProvider]    (plug in any other data source)

    PitchControlAnalyzer
        ├── takes any BaseDataProvider
        ├── computes velocities
        ├── runs pitch control model (Spearman / Fernandez & Bornn)
        └── produces:  animation / PNG frames / CSV reports
"""

from __future__ import annotations

import abc
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all OS)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FFMpegWriter, PillowWriter
from scipy.stats import multivariate_normal
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
PITCH_LENGTH : float = 105.0   # metres
PITCH_WIDTH  : float = 68.0    # metres
GRID_NX      : int   = 51      # grid columns
GRID_NY      : int   = 45      # grid rows
DEFAULT_FPS  : int   = 25      # frames per second

MAX_PLAYER_SPEED : float = 13.0   # m/s  (eq. 18 denominator)
MIN_RADIUS       : float = 4.0    # m    (Figure 9)
MAX_RADIUS       : float = 10.0   # m    (Figure 9)
MIN_SY           : float = 0.1    # avoids singular covariance


# ══════════════════════════════════════════════════════════════
#  STANDARD DATA STRUCTURES
# ══════════════════════════════════════════════════════════════
@dataclass
class PlayerFrame:
    """Position of one player in one frame (velocity added later)."""
    player_id : str
    name      : str
    team      : str                    # 'home' | 'away'
    x         : float                  # metres (0 → PITCH_LENGTH)
    y         : float                  # metres (0 → PITCH_WIDTH)
    speed     : float = 0.0            # scalar speed (m/s) if available
    vx        : float = field(default=0.0, init=False)
    vy        : float = field(default=0.0, init=False)

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vx, self.vy])


@dataclass
class TrackingFrame:
    """All positions in one frame."""
    frame_id        : int
    ball_x          : Optional[float]
    ball_y          : Optional[float]
    players         : List[PlayerFrame] = field(default_factory=list)
    # DFL BallPossession: 1=home, 2=away, 0/None=no possession
    ball_possession : Optional[int]     = None
    # DFL BallStatus: 1=alive, 0=dead
    ball_status     : Optional[int]     = None
    # DFL GameSection: 'firstHalf' | 'secondHalf'
    game_section    : str               = 'firstHalf'
    # Ball speed in km/h between this frame and the previous one (filled by _compute_velocities)
    ball_speed_kmh  : float             = 0.0

    @property
    def ball_pos(self) -> Optional[np.ndarray]:
        if self.ball_x is None or self.ball_y is None:
            return None
        return np.array([self.ball_x, self.ball_y])

    @property
    def possessing_team(self) -> Optional[str]:
        """Return 'home', 'away', or None when possession is unknown/contested."""
        if self.ball_possession == 1:
            return 'home'
        if self.ball_possession == 2:
            return 'away'
        return None


# ══════════════════════════════════════════════════════════════
#  ABSTRACT DATA PROVIDER
# ══════════════════════════════════════════════════════════════
class BaseDataProvider(abc.ABC):
    """
    Interface that any data source must implement.
    Subclass this to support StatsPerform, Second Spectrum, CSV, etc.
    """

    @abc.abstractmethod
    def load(self,
             start_frame: Optional[int] = None,
             end_frame  : Optional[int] = None
             ) -> Dict[int, TrackingFrame]:
        """
        Load tracking data and return a dict keyed by frame number.
        Coordinates must already be in metres, origin at pitch corner (0,0).
        """

    @property
    @abc.abstractmethod
    def fps(self) -> int:
        """Frame rate of the tracking data."""

    @property
    @abc.abstractmethod
    def home_team_id(self) -> str:
        """Identifier of the home team."""

    @property
    @abc.abstractmethod
    def away_team_id(self) -> str:
        """Identifier of the away team."""


# ══════════════════════════════════════════════════════════════
#  DFL DATA PROVIDER
# ══════════════════════════════════════════════════════════════
class DFLDataProvider(BaseDataProvider):
    """
    Reads DFL (Bundesliga) open-data XML files:
      • DFL_02_01_matchinformation_...xml  →  player names & teams
      • DFL_04_03_positions_raw_observed_...xml  →  tracking positions

    DFL coordinates are centred at (0, 0).  We shift them so that
    the bottom-left corner of the pitch becomes (0, 0).
    """

    _FPS = 25

    def __init__(self, data_dir: str, match_id: str,
                 direction_json: Optional[str] = None):
        self.data_dir = data_dir
        self.match_id = match_id
        self._player_meta  : Dict[str, dict] = {}
        self._home_team_id : str = ''
        self._away_team_id : str = ''
        self._play_direction: Dict[str, dict] = {}
        self._parse_match_info()
        if direction_json:
            self.load_play_direction(direction_json)

    # ── public properties ───────────────────────────────────

    @property
    def fps(self) -> int:
        return self._FPS

    @property
    def home_team_id(self) -> str:
        return self._home_team_id

    @property
    def away_team_id(self) -> str:
        return self._away_team_id

    @property
    def player_meta(self) -> Dict[str, dict]:
        return self._player_meta

    # ── play direction ──────────────────────────────────────

    def load_play_direction(self, json_path: str) -> None:
        """
        Load play direction JSON file.
        Expected format:
            { "J03WMX": { "Home": { "firstHalf": "right_to_left",
                                    "secondHalf": "left_to_right" } } }
        """
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        match_short = self.match_id.replace('DFL-MAT-', '')
        self._play_direction = data.get(match_short, {})
        if self._play_direction:
            print(f'[DFL] Play direction loaded for match {match_short}')
        else:
            print(f'[WARN] No play direction found for match {match_short}')

    def get_attack_direction(self, team: str, game_section: str) -> str:
        """
        Return 'left_to_right' or 'right_to_left' for the given team
        and game section.  Falls back to 'left_to_right' if no direction
        data is available.
        """
        if not self._play_direction:
            return 'left_to_right'
        home_dir = (self._play_direction
                    .get('Home', {})
                    .get(game_section, 'left_to_right'))
        if team == 'home':
            return home_dir
        else:
            return ('right_to_left'
                    if home_dir == 'left_to_right'
                    else 'left_to_right')

    # ── loading ─────────────────────────────────────────────

    # Sentinel PersonId used by DFL for the ball object
    _BALL_PERSON_ID = 'BALL'

    def load(self,
             start_frame: Optional[int] = None,
             end_frame  : Optional[int] = None
             ) -> Dict[int, TrackingFrame]:
        """
        Parse DFL position XML.

        Real DFL structure
        ------------------
        Each <FrameSet> belongs to **one** player or the ball, identified by
        the FrameSet-level attributes:

            <FrameSet GameSection="firstHalf"
                      MatchId="..."
                      TeamId="DFL-CLU-000008"
                      PersonId="DFL-OBJ-0027AX">
                <Frame N="10000" T="..." X="6.90" Y="5.12"
                       D="0.00" S="0.00" A="0.00" M="1"/>
                ...
            </FrameSet>

        The ball FrameSet has PersonId == _BALL_PERSON_ID (or no TeamId).
        Coordinates are in metres, already relative to the pitch centre
        (origin at centre circle).  We shift them to bottom-left origin.
        """
        path = self._find_file('positions_raw')
        print(f'[DFL] Parsing positions: {os.path.basename(path)}')
        tree = ET.parse(path)
        root = tree.getroot()

        # ── detect XML namespace (e.g. xmlns="dfl:...") ─────────
        # When a namespace is present ET prefixes every tag: {ns}FrameSet
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'
            print(f'[DFL] XML namespace detected: {ns}')

        tag_frameset = f'{ns}FrameSet'
        tag_frame    = f'{ns}Frame'

        frames: Dict[int, TrackingFrame] = {}

        def _in_range(n: int) -> bool:
            if start_frame is not None and n < start_frame:
                return False
            if end_frame is not None and n > end_frame:
                return False
            return True

        for frameset in root.iter(tag_frameset):
            person_id    = frameset.get('PersonId', '')
            team_id      = frameset.get('TeamId',   '')
            game_section = frameset.get('GameSection', 'firstHalf')

            # ── skip referees ─────────────────────────────────────────
            if team_id.lower() == 'referee':
                continue

            # ── determine whether this FrameSet is ball or player ──
            # DFL uses TeamId="BALL" (not empty) to mark the ball FrameSet.
            is_ball = (
                team_id.upper() == 'BALL'
                or person_id.upper() == 'BALL'
            )

            # player meta lookup.
            # matchinformation PlayerId == positions PersonId in well-formed
            # DFL files.  If not found in meta, fall back to TeamId for the
            # home/away role so all FrameSets still contribute to pitch control.
            meta      = self._player_meta.get(person_id, {})
            team_role = meta.get('team', '')
            if not team_role and team_id:
                if team_id == self._home_team_id:
                    team_role = 'home'
                elif team_id == self._away_team_id:
                    team_role = 'away'
                else:
                    team_role = 'unknown'

            for frame_elem in frameset.findall(tag_frame):
                n = int(frame_elem.get('N', 0))
                if not _in_range(n):
                    continue

                # Convert from centre-origin to bottom-left origin
                x = float(frame_elem.get('X', 0)) + PITCH_LENGTH / 2
                y = float(frame_elem.get('Y', 0)) + PITCH_WIDTH  / 2

                if is_ball:
                    if n not in frames:
                        frames[n] = TrackingFrame(frame_id=n,
                                                  ball_x=None, ball_y=None)
                    frames[n].ball_x       = x
                    frames[n].ball_y       = y
                    frames[n].game_section = game_section
                    raw_poss = frame_elem.get('BallPossession')
                    raw_stat = frame_elem.get('BallStatus')
                    frames[n].ball_possession = int(raw_poss) if raw_poss is not None else None
                    frames[n].ball_status     = int(raw_stat) if raw_stat is not None else None
                else:
                    if n not in frames:
                        frames[n] = TrackingFrame(frame_id=n,
                                                  ball_x=None, ball_y=None)
                    frames[n].game_section = game_section
                    speed = float(frame_elem.get('S', 0))
                    frames[n].players.append(PlayerFrame(
                        player_id = person_id,
                        name      = meta.get('name', person_id),
                        team      = team_role,
                        x         = x,
                        y         = y,
                        speed     = speed,
                    ))

        print(f'[DFL] {len(frames)} frames loaded.')
        return frames

    def frame_range(self) -> tuple:
        """
        Return (min_frame, max_frame) by scanning the positions XML.
        Useful to know valid start/end values before calling analyze().
        """
        path = self._find_file('positions_raw')
        tree = ET.parse(path)
        root = tree.getroot()
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'
        tag_frameset = f'{ns}FrameSet'
        tag_frame    = f'{ns}Frame'
        ns_values = []
        for frameset in root.iter(tag_frameset):
            for frame_elem in frameset.findall(tag_frame):
                n = frame_elem.get('N')
                if n is not None:
                    ns_values.append(int(n))
        if not ns_values:
            return (0, 0)
        return (min(ns_values), max(ns_values))

    # ── private helpers ─────────────────────────────────────

    def _find_file(self, pattern: str) -> str:
        candidates = [
            f for f in os.listdir(self.data_dir)
            if pattern in f and self.match_id in f
        ]
        if not candidates:
            raise FileNotFoundError(
                f'No file matching pattern "{pattern}" and '
                f'match_id "{self.match_id}" in {self.data_dir}')
        return os.path.join(self.data_dir, candidates[0])

    def _parse_match_info(self):
        """
        DFL matchinformation structure:
          <General HomeTeamId="..." GuestTeamId="..." />
          <Teams>
            <Team TeamId="..." Role="home|guest">
              <Players>
                <Player PersonId="..." FirstName="..." LastName="..."
                        Shortname="..." PlayingPosition="..." Starting="true|false" />
        """
        path = self._find_file('matchinformation')
        print(f'[DFL] Parsing match info: {os.path.basename(path)}')
        tree = ET.parse(path)
        root = tree.getroot()

        # ── detect namespace ──────────────────────────────────────
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'

        def _itag(tag):
            return f'{ns}{tag}'

        # ── read home/guest team IDs from <General> ───────────────
        for gen in root.iter(_itag('General')):
            self._home_team_id = (gen.get('HomeTeamId') or
                                  gen.get('homeTeamId') or '')
            self._away_team_id = (gen.get('GuestTeamId') or
                                  gen.get('guestTeamId') or
                                  gen.get('AwayTeamId')  or '')
            break   # only one <General>

        # ── parse players from <Teams><Team><Players><Player> ─────
        for team_elem in root.iter(_itag('Team')):
            team_id = (team_elem.get('TeamId') or team_elem.get('teamId') or '')
            role_raw = (team_elem.get('Role')  or team_elem.get('role')  or '').lower()

            if role_raw == 'home':
                team_role = 'home'
            elif role_raw in ('guest', 'away', 'visitor'):
                team_role = 'away'
            else:
                # fallback: compare TeamId to what we read from <General>
                if team_id and team_id == self._home_team_id:
                    team_role = 'home'
                elif team_id and team_id == self._away_team_id:
                    team_role = 'away'
                else:
                    team_role = 'unknown'

            for p in team_elem.iter(_itag('Player')):
                pid = (p.get('PersonId') or p.get('PlayerId') or
                       p.get('personId') or p.get('id') or '')
                if not pid:
                    continue
                # build display name: prefer Shortname, else FirstName + LastName
                shortname = p.get('Shortname', p.get('shortname', ''))
                firstname = p.get('FirstName', p.get('firstName', ''))
                lastname  = p.get('LastName',  p.get('lastName',  ''))
                name = shortname or f'{firstname} {lastname}'.strip() or pid

                self._player_meta[pid] = {
                    'name'    : name,
                    'team'    : team_role,
                    'team_id' : team_id,
                    'position': p.get('PlayingPosition', ''),
                    'starting': p.get('Starting', 'false').lower() == 'true',
                }

        print(f'[DFL] {len(self._player_meta)} players found  '
              f'(home={self._home_team_id}, away={self._away_team_id})')


# ══════════════════════════════════════════════════════════════
#  PITCH CONTROL MATHEMATICS
# ══════════════════════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function → [0, 1]."""
    return 1.0 / (1.0 + np.exp(-x))


def _compute_mu(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """
    Mean of the player's Gaussian: position shifted 0.5 m in movement
    direction (Equation 21).  If stationary, centre = position.
    """
    norm = np.linalg.norm(vel)
    if norm < 1e-6:
        return pos.copy()
    return pos + (vel / norm) * 0.5


def _compute_radius(dist_to_ball: float) -> float:
    """
    Influence radius as a function of distance to ball (Figure 9).
    Clamped to [MIN_RADIUS, MAX_RADIUS].
    """
    r = MIN_RADIUS + 6.0 * (1.0 - np.exp(-dist_to_ball / 10.0))
    return float(np.clip(r, MIN_RADIUS, MAX_RADIUS))


def _create_rotation_matrix(vel: np.ndarray) -> np.ndarray:
    """
    2-D rotation matrix aligned with velocity direction (Equation 16).
    Returns identity if player is stationary.
    """
    if np.linalg.norm(vel) < 1e-6:
        return np.eye(2)
    theta = np.arctan2(vel[1], vel[0])
    c, s  = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _create_scaling_matrix(vel: np.ndarray, radius: float) -> np.ndarray:
    """
    Scaling matrix S from Equations 18-19.
    The ellipse is stretched in the direction of motion (sx) and
    contracted perpendicularly (sy).  sy is floored at MIN_SY to
    prevent a singular covariance matrix.
    """
    speed = np.linalg.norm(vel)
    srat  = speed**2 / MAX_PLAYER_SPEED**2          # Equation 18
    sx    = (radius + radius * srat) / 2.0           # Equation 19
    sy    = max((radius - radius * srat) / 2.0, MIN_SY)
    return np.array([[sx, 0.0], [0.0, sy]])


def _compute_covariance(vel         : np.ndarray,
                         dist_to_ball: float,
                         fixed_radius: bool = False) -> np.ndarray:
    """
    Full covariance matrix Σ = R S Sᵀ Rᵀ  (Equation 20).
    Accounts for player velocity direction and distance to ball.

    fixed_radius : when True (ball travelling above the speed threshold),
                   skip the distance-to-ball radius scaling and use
                   MIN_RADIUS for every player.  This prevents a fast
                   through-ball from artificially shrinking the influence
                   radius of the players it lobs over.
    """
    R      = _create_rotation_matrix(vel)
    radius = MAX_RADIUS if fixed_radius else _compute_radius(dist_to_ball)
    S      = _create_scaling_matrix(vel, radius)
    return R @ S @ S @ R.T


def _player_influence(grid        : np.ndarray,
                       pos         : np.ndarray,
                       vel         : np.ndarray,
                       ball_pos    : np.ndarray,
                       fixed_radius: bool = False) -> np.ndarray:
    """
    Influence of one player on every grid point (Equations 12-13).

    I_i(p, t) = f_i(p, t) / f_i(p_i(t), t)

    Returns array of shape (N,) with values in [0, 1].

    fixed_radius : forwarded to _compute_covariance.  When True,
                   the influence ellipse is built with MIN_RADIUS
                   regardless of the player's distance to the ball.
    """
    mu   = _compute_mu(pos, vel)
    dist = float(np.linalg.norm(pos - ball_pos))
    cov  = _compute_covariance(vel, dist, fixed_radius=fixed_radius)

    pdf_grid   = multivariate_normal.pdf(grid, mean=mu, cov=cov)
    pdf_player = multivariate_normal.pdf(pos,  mean=mu, cov=cov)

    # avoid division by zero (can happen if cov is near-singular)
    return pdf_grid / max(pdf_player, 1e-12)


def compute_pitch_control(grid        : np.ndarray,
                           att_players : List[PlayerFrame],
                           def_players : List[PlayerFrame],
                           ball_pos    : np.ndarray,
                           fixed_radius: bool = False
                           ) -> np.ndarray:
    """
    Team pitch control surface (Equation 2):
        PC(p, t) = σ( Σ I_att(p,t) − Σ I_def(p,t) )

    Returns array of shape (N,) in [0, 1].
        1  →  attacking team fully controls this point
        0  →  defending team fully controls this point
       0.5 →  contested

    fixed_radius : when True (ball faster than threshold), use MIN_RADIUS
                   for all influence ellipses (see _compute_covariance).
    """
    sum_att = np.zeros(len(grid))
    sum_def = np.zeros(len(grid))

    for p in att_players:
        sum_att += _player_influence(grid, p.pos, p.vel, ball_pos, fixed_radius)
    for p in def_players:
        sum_def += _player_influence(grid, p.pos, p.vel, ball_pos, fixed_radius)

    return _sigmoid(sum_att - sum_def)


def compute_pitch_control_single(grid        : np.ndarray,
                                  players     : List[PlayerFrame],
                                  ball_pos    : np.ndarray,
                                  fixed_radius: bool = False
                                  ) -> np.ndarray:
    """
    Pitch control surface for ONE team in isolation:
        PC_single(p, t) = σ( Σ I_team(p, t) )

    Useful to see the raw influence footprint of a single team without
    the opponent pulling values toward 0.

    Returns array of shape (N,) in [0, 1].
        1  →  strong control by this team
       0.5 →  moderate influence
        0  →  no influence
    """
    sum_inf = np.zeros(len(grid))
    for p in players:
        sum_inf += _player_influence(grid, p.pos, p.vel, ball_pos, fixed_radius)
    return _sigmoid(sum_inf)


def compute_ball_zone_control(grid    : np.ndarray,
                               PC      : np.ndarray,
                               centre  : np.ndarray,
                               radius  : float = 6.0
                               ) -> dict:
    """
    Aggregate pitch control inside a circular zone of given *radius*
    around *centre* (metres).

    Parameters
    ----------
    grid   : (N, 2) array of grid point coordinates
    PC     : (N,)   pitch-control values in [0, 1]  (att team = 1)
    centre : (2,)   [x, y] centre of the zone (ball or player position)
    radius : float  radius of the zone in metres (default 6 m)

    Returns
    -------
    dict with keys:
        'pc_att_zone'  – mean PC (att team) inside the zone  [0-1]
        'pc_def_zone'  – mean PC (def team) inside the zone  [0-1]
        'n_grid_pts'   – number of grid points inside the zone
    """
    dists = np.linalg.norm(grid - centre, axis=1)
    mask  = dists <= radius
    n_pts = int(mask.sum())

    if n_pts == 0:
        return {'pc_att_zone': float('nan'),
                'pc_def_zone': float('nan'),
                'n_grid_pts' : 0}

    pc_att_zone = float(PC[mask].mean())
    return {
        'pc_att_zone' : round(pc_att_zone,        4),
        'pc_def_zone' : round(1.0 - pc_att_zone,  4),
        'n_grid_pts'  : n_pts,
    }


# ══════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════

def _draw_pitch(ax: plt.Axes,
                line_color: str = 'black',
                lw: float = 1.5) -> None:
    """Draw standard football pitch markings."""
    ax.set_facecolor('#3a7d44')
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect('equal')
    ax.axis('off')
    kw = dict(color=line_color, linewidth=lw)

    # boundary
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0],
            [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], **kw)
    # halfway line
    ax.plot([PITCH_LENGTH / 2]*2, [0, PITCH_WIDTH], **kw)
    # centre circle
    ax.add_patch(plt.Circle((PITCH_LENGTH/2, PITCH_WIDTH/2),
                             9.15, fill=False, **kw))
    # penalty areas
    for x0 in [0, PITCH_LENGTH - 16.5]:
        ax.add_patch(mpatches.Rectangle(
            (x0, (PITCH_WIDTH - 40.32) / 2), 16.5, 40.32,
            fill=False, **kw))
    # goal boxes
    for x0 in [0, PITCH_LENGTH - 5.5]:
        ax.add_patch(mpatches.Rectangle(
            (x0, (PITCH_WIDTH - 18.32) / 2), 5.5, 18.32,
            fill=False, **kw))


def _render_frame(ax        : plt.Axes,
                  PC_2d     : np.ndarray,
                  frame     : TrackingFrame,
                  frame_id  : int,
                  att_team  : str,
                  target_player: Optional[str] = None,
                  cmap      : str = 'RdYlGn_r') -> None:
    """Render one pitch-control frame onto *ax* (clears ax first)."""
    ax.cla()
    _draw_pitch(ax)

    ax.imshow(PC_2d,
              extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
              origin='lower', cmap=cmap, alpha=0.75,
              vmin=0, vmax=1, aspect='auto')

    for p in frame.players:
        if target_player is not None and p.name == target_player:
            color = '#0000cd'
        elif p.team == att_team:
            color = "#99BAFC"
        else:
            color = '#FF1A00'

        ax.scatter(p.x, p.y, c=color, s=70, zorder=5,
                   edgecolors='black', linewidths=0.5)
        ax.annotate(p.name, (p.x, p.y),
                    fontsize=5, color='white', zorder=6,
                    xytext=(3, 3), textcoords='offset points')

    if frame.ball_pos is not None:
        ax.scatter(*frame.ball_pos, c='yellow', s=100, zorder=7,
                   edgecolors='black', linewidths=1)

    pc_att  = float(PC_2d.mean())
    pc_def  = 1 - pc_att
    ctrl    = att_team if pc_att >= 0.5 else ('away' if att_team == 'home' else 'home')
    poss    = frame.possessing_team
    poss_str = f'  |  possession: {poss.upper()}' if poss else '  |  possession: —'
    pct_str  = f'{pc_att*100:.1f}% att  /  {pc_def*100:.1f}% def'
    ax.set_title(
        f'Pitch Control  |  frame {frame_id}{poss_str}  |  {ctrl.upper()} controls ({pct_str})',
        fontsize=9)


# ══════════════════════════════════════════════════════════════
#  MAIN ANALYSER CLASS
# ══════════════════════════════════════════════════════════════

class PitchControlAnalyzer:
    """
    Source-agnostic pitch control analyser.

    Parameters
    ----------
    provider    : any BaseDataProvider subclass
    output_dir  : root folder for all outputs
    att_team    : 'home' or 'away'  (the attacking / reference team)
    """

    def __init__(self,
                 provider  : BaseDataProvider,
                 output_dir: str,
                 att_team  : str = 'home'):
        self.provider   = provider
        self.output_dir = output_dir
        self.att_team   = att_team
        self.def_team   = 'away' if att_team == 'home' else 'home'
        self._grid      = self._build_grid()
        os.makedirs(output_dir, exist_ok=True)

    # ── public API ──────────────────────────────────────────

    def analyze(self,
                start_frame        : int,
                end_frame          : int,
                save_animation     : bool          = True,
                save_frame_images  : bool          = False,
                save_csv           : bool          = True,
                animation_format   : str           = 'gif',
                # ── snapshot / zone-control options ──────────────────
                save_snapshot_csv   : bool          = True,
                ball_control_radius : float         = 6.0,
                player_zone_radius  : float         = 6.0,
                target_player       : Optional[str] = None,
                # ── ball-speed radius freeze ──────────────────────────
                ball_speed_threshold: float         = 35.0,
                ) -> pd.DataFrame:
        """
        Run pitch control over [start_frame, end_frame] and save outputs.

        New parameters
        --------------
        save_snapshot_csv   : write snapshot_summary.csv with one row each for
                              start, mid, and end frame.  Each row contains:
                                - pc_home_only / pc_away_only  (single-team PC)
                                - pc_combined_home / pc_combined_away  (joint PC)
                                - ball_zone_pc_home/away  (zone around the ball)
                                - player_zone_pc_home/away  (zone around target_player)
        ball_control_radius : radius in metres for the ball/player zone (default 6 m)
        player_zone_radius  : radius in metres for the target_player zone (default 6 m)
        target_player       : name of the player to analyse zone control around
                              (partial match, case-insensitive).  None → skipped.

        Returns a DataFrame with per-frame, per-player metrics.
        """
        frames = self.provider.load(start_frame, end_frame)
        if not frames:
            print('[WARN] No frames loaded — check start_frame/end_frame '
                  'range and that the XML file path is correct.')
            return pd.DataFrame()

        frames = self._compute_velocities(frames, fps=self.provider.fps)

        # Use actual frame IDs (not the requested range) for the folder name
        sorted_ids = sorted(frames.keys())
        actual_start = sorted_ids[0]  if sorted_ids else start_frame
        actual_end   = sorted_ids[-1] if sorted_ids else end_frame

        run_name   = f'frames_{actual_start}_{actual_end}_{target_player or "all"}'
        run_dir    = os.path.join(self.output_dir, run_name)
        img_dir    = os.path.join(run_dir, 'frames')
        os.makedirs(run_dir,  exist_ok=True)   # always create before writer.setup
        if save_frame_images:
            os.makedirs(img_dir, exist_ok=True)

        records          = []   # player-level metrics
        frame_records    = []   # one row per frame
        snapshot_records = []   # start / mid / end snapshots

        # ── identify start / mid / end snapshot frame ids ────
        mid_idx      = len(sorted_ids) // 2
        snapshot_ids = {
            'start': sorted_ids[0],
            'mid'  : sorted_ids[mid_idx],
            'end'  : sorted_ids[-1],
        }
        print(f'[INFO] Snapshot frames → '
              f'start={snapshot_ids["start"]}  '
              f'mid={snapshot_ids["mid"]}  '
              f'end={snapshot_ids["end"]}')

        # ── pre-allocate figure for animation ───────────────
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.tight_layout()
        writer  = None

        if save_animation:
            anim_path = os.path.join(run_dir, f'pitch_control.{animation_format}')
            if animation_format == 'mp4':
                writer = FFMpegWriter(fps=self.provider.fps)
            else:
                writer = PillowWriter(fps=min(self.provider.fps, 10))
            writer.setup(fig, anim_path, dpi=120)

        for fid in tqdm(sorted_ids, desc='Analysing frames'):
            frame   = frames[fid]
            ball    = frame.ball_pos
            if ball is None:
                continue

            # ── determine att/def from ball possession ───────────
            # Fall back to the analyser's default att_team when possession
            # is unknown (contested, dead ball, no data).
            poss_team = frame.possessing_team   # 'home', 'away', or None
            if poss_team is not None:
                frame_att = poss_team
                frame_def = 'away' if poss_team == 'home' else 'home'
            else:
                frame_att = self.att_team
                frame_def = self.def_team

            att = [p for p in frame.players if p.team == frame_att]
            dff = [p for p in frame.players if p.team == frame_def]

            # ── ball-speed radius freeze ──────────────────────────
            ball_fast    = frame.ball_speed_kmh > ball_speed_threshold
            ball_spd_kmh = round(frame.ball_speed_kmh, 2)

            PC      = compute_pitch_control(
                self._grid, att, dff, ball, fixed_radius=ball_fast)
            PC_2d   = PC.reshape(GRID_NY, GRID_NX)

            # ── frame-level pitch control metrics ────────────────
            pc_att  = float(PC.mean())          # att team control [0-1]
            pc_def  = float(1 - pc_att)
            ctrl    = frame_att if pc_att >= 0.5 else frame_def

            # zone breakdown: thirds oriented by attack direction
            nx      = GRID_NX
            third   = nx // 3
            PC_2d_t = PC.reshape(GRID_NY, GRID_NX)

            direction = self.provider.get_attack_direction(
                frame_att, frame.game_section)

            if direction == 'left_to_right':
                # frame_att attacks toward x=105 (right)
                # left third  = their defensive third
                # right third = their attacking third
                pc_def_third = float(PC_2d_t[:, :third].mean())
                pc_mid_third = float(PC_2d_t[:, third:2*third].mean())
                pc_att_third = float(PC_2d_t[:, 2*third:].mean())
            else:
                # frame_att attacks toward x=0 (left) → invert
                pc_def_third = float(PC_2d_t[:, 2*third:].mean())
                pc_mid_third = float(PC_2d_t[:, third:2*third].mean())
                pc_att_third = float(PC_2d_t[:, :third].mean())

            # ── single-team pitch control (home / away always) ───
            home_players = [p for p in frame.players if p.team == 'home']
            away_players = [p for p in frame.players if p.team == 'away']

            PC_home_only = compute_pitch_control_single(
                self._grid, home_players, ball, fixed_radius=ball_fast)
            PC_away_only = compute_pitch_control_single(
                self._grid, away_players, ball, fixed_radius=ball_fast)

            # Combined PC expressed as home / away fractions
            # (PC already gives att=1 / def=0; we convert to home/away)
            if frame_att == 'home':
                pc_combined_home = round(pc_att, 4)
                pc_combined_away = round(pc_def, 4)
            else:
                pc_combined_home = round(pc_def, 4)
                pc_combined_away = round(pc_att, 4)

            pc_home_only_mean = round(float(PC_home_only.mean()), 4)
            pc_away_only_mean = round(float(PC_away_only.mean()), 4)

            # ── zone control: ball radius ─────────────────────
            ball_zone = compute_ball_zone_control(
                self._grid, PC, ball, radius=ball_control_radius)
            # express as home / away
            if frame_att == 'home':
                ball_zone_home = ball_zone['pc_att_zone']
                ball_zone_away = ball_zone['pc_def_zone']
            else:
                ball_zone_home = ball_zone['pc_def_zone']
                ball_zone_away = ball_zone['pc_att_zone']

            # ── zone control: target player radius ────────────
            player_zone_home = float('nan')
            player_zone_away = float('nan')
            target_player_found_name = None
            if target_player is not None:
                query = target_player.lower()
                matched = [p for p in frame.players
                           if query in p.name.lower()]
                if matched:
                    tp = matched[0]
                    target_player_found_name = tp.name
                    pz = compute_ball_zone_control(
                        self._grid, PC, tp.pos, radius=player_zone_radius)
                    if frame_att == 'home':
                        player_zone_home = pz['pc_att_zone']
                        player_zone_away = pz['pc_def_zone']
                    else:
                        player_zone_home = pz['pc_def_zone']
                        player_zone_away = pz['pc_att_zone']

            # ── snapshot record (start / mid / end frames only) ──
            snap_label = None
            for label, sid in snapshot_ids.items():
                if fid == sid:
                    snap_label = label
                    break
            if snap_label is not None:
                snap = {
                    'snapshot'              : snap_label,
                    'frame_id'              : fid,
                    'game_section'          : frame.game_section,
                    'ball_x'               : round(float(ball[0]), 2),
                    'ball_y'               : round(float(ball[1]), 2),
                    'possessing_team'      : poss_team if poss_team else 'unknown',
                    'att_team'             : frame_att,
                    'ball_speed_kmh'       : ball_spd_kmh,
                    'radius_frozen'        : ball_fast,
                    # single-team PC (each team alone, ignoring the other)
                    'pc_home_only'         : pc_home_only_mean,
                    'pc_away_only'         : pc_away_only_mean,
                    # combined PC (both teams, expressed as home/away share)
                    'pc_combined_home'     : pc_combined_home,
                    'pc_combined_away'     : pc_combined_away,
                    # zone around the ball
                    f'ball_zone_{ball_control_radius}m_home': ball_zone_home,
                    f'ball_zone_{ball_control_radius}m_away': ball_zone_away,
                }
                if target_player is not None:
                    tp_label = (target_player_found_name or target_player).replace(' ', '_')
                    snap[f'player_zone_{tp_label}_home'] = player_zone_home
                    snap[f'player_zone_{tp_label}_away'] = player_zone_away
                snapshot_records.append(snap)

            # ── collect metrics ───────────────────────────────
            for p in att + dff:
                inf = _player_influence(self._grid, p.pos, p.vel, ball)
                records.append({
                    'frame_id'           : fid,
                    'game_section'       : frame.game_section,
                    'player_id'          : p.player_id,
                    'name'               : p.name,
                    'team'               : p.team,
                    'x'                  : p.x,
                    'y'                  : p.y,
                    'speed_ms'           : float(np.linalg.norm(p.vel)),
                    'mean_influence'     : float(inf.mean()),
                    'max_influence'      : float(inf.max()),
                    'team_pc_mean'       : float(PC.mean()) if p.team == frame_att else float(1 - PC.mean()),
                    # frame-level columns (same for every player in this frame)
                    'possessing_team'    : poss_team if poss_team else 'unknown',
                    'att_team'           : frame_att,
                    'pc_att_team'        : round(pc_att,  4),
                    'pc_def_team'        : round(pc_def,  4),
                    'controlling_team'   : ctrl,
                    'pc_def_third'       : round(pc_def_third, 4),
                    'pc_mid_third'       : round(pc_mid_third, 4),
                    'pc_att_third'       : round(pc_att_third, 4),
                })

            frame_records.append({
                'frame_id'         : fid,
                'game_section'     : frame.game_section,
                'possessing_team'  : poss_team if poss_team else 'unknown',
                'ball_status'      : frame.ball_status,
                'pc_att_team'      : round(pc_att, 4),
                'pc_def_team'      : round(pc_def, 4),
                'controlling_team' : ctrl,
                'pc_def_third'     : round(pc_def_third,  4),
                'pc_mid_third'     : round(pc_mid_third,  4),
                'pc_att_third'     : round(pc_att_third,  4),
                'ball_x'           : round(float(ball[0]), 2),
                'ball_y'           : round(float(ball[1]), 2),
                'n_att_players'    : len(att),
                'n_def_players'    : len(dff),
                'ball_speed_kmh'   : ball_spd_kmh,
                'radius_frozen'    : ball_fast,
            })

            # ── render ────────────────────────────────────────
            _render_frame(ax, PC_2d, frame, fid, frame_att, target_player=target_player)

            if save_frame_images:
                fig.savefig(os.path.join(img_dir, f'frame_{fid:07d}.png'),
                            dpi=100)

            if save_animation and writer:
                writer.grab_frame()

        if save_animation and writer:
            try:
                writer.finish()
                print(f'[OUT] Animation → {anim_path}')
            except (IndexError, Exception) as e:
                print(f'[WARN] Animation not saved (no frames rendered): {e}')

        plt.close(fig)

        # ── build & save DataFrame ───────────────────────────
        df          = pd.DataFrame(records)
        df_frame    = pd.DataFrame(frame_records)
        df_snapshot = pd.DataFrame(snapshot_records)

        if save_csv:
            if not df.empty:
                self._save_csv(df, run_dir)
            if not df_frame.empty:
                fc_path = os.path.join(run_dir, 'frame_control.csv')
                df_frame.to_csv(fc_path, index=False)
                print(f'[OUT] Frame control CSV → {fc_path}')

        if save_snapshot_csv and not df_snapshot.empty:
            snap_path = os.path.join(run_dir, 'snapshot_summary.csv')
            df_snapshot.to_csv(snap_path, index=False)
            print(f'[OUT] Snapshot summary CSV → {snap_path}')
            print('\n── Snapshot summary ───────────────────')
            print(df_snapshot.to_string(index=False))
            print('─' * 55)

        self._print_summary(df, run_dir)
        return df

    # ── private helpers ─────────────────────────────────────

    @staticmethod
    def _build_grid() -> np.ndarray:
        """Build flat (GRID_NX × GRID_NY, 2) coordinate array."""
        xs = np.linspace(0, PITCH_LENGTH, GRID_NX)
        ys = np.linspace(0, PITCH_WIDTH,  GRID_NY)
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack([xx.ravel(), yy.ravel()])

    @staticmethod
    def _compute_velocities(frames: Dict[int, TrackingFrame],
                             fps: int = DEFAULT_FPS
                             ) -> Dict[int, TrackingFrame]:
        """
        Estimate velocity vectors from consecutive positions.
        Boundary frames receive zero velocity.
        Modifies frames in-place and returns them.
        """
        dt      = 1.0 / fps
        ids     = sorted(frames.keys())
        prev_pos: Dict[str, np.ndarray] = {}

        prev_ball: Optional[np.ndarray] = None

        for i, fid in enumerate(ids):
            frame = frames[fid]

            # -- ball speed (m/s -> km/h) --
            ball = frame.ball_pos
            if ball is not None and prev_ball is not None:
                ball_dist_m = float(np.linalg.norm(ball - prev_ball))
                frame.ball_speed_kmh = (ball_dist_m / dt) * 3.6
            else:
                frame.ball_speed_kmh = 0.0
            prev_ball = ball

            for p in frame.players:
                if p.player_id in prev_pos and i > 0:
                    delta  = p.pos - prev_pos[p.player_id]
                    p.vx, p.vy = float(delta[0] / dt), float(delta[1] / dt)
                else:
                    p.vx = p.vy = 0.0
                prev_pos[p.player_id] = p.pos.copy()

        return frames

    def _save_csv(self, df: pd.DataFrame, run_dir: str) -> None:
        """Save per-frame metrics and per-player aggregated summary."""
        # full frame-by-frame data
        detail_path = os.path.join(run_dir, 'metrics_detail.csv')
        df.to_csv(detail_path, index=False)
        print(f'[OUT] Detail CSV  → {detail_path}')

        # aggregated per player
        agg = (df.groupby(['player_id', 'name', 'team'])
                 .agg(
                     frames_tracked  = ('frame_id',       'count'),
                     mean_influence  = ('mean_influence',  'mean'),
                     max_influence   = ('max_influence',   'max'),
                     mean_speed_ms   = ('speed_ms',        'mean'),
                     max_speed_ms    = ('speed_ms',        'max'),
                 )
                 .reset_index()
                 .sort_values('mean_influence', ascending=False))

        summary_path = os.path.join(run_dir, 'metrics_summary.csv')
        agg.to_csv(summary_path, index=False)
        print(f'[OUT] Summary CSV → {summary_path}')

    @staticmethod
    def _print_summary(df: pd.DataFrame, run_dir: str) -> None:
        """Pretty-print a leaderboard to stdout and save as .txt."""
        if df.empty:
            print('[WARN] No data to summarise.')
            return

        agg = (df.groupby(['name', 'team'])
                 .agg(mean_inf=('mean_influence', 'mean'),
                      max_speed=('speed_ms', 'max'))
                 .reset_index()
                 .sort_values('mean_inf', ascending=False))

        lines = [
            '═' * 58,
            f'  PITCH CONTROL SUMMARY',
            '═' * 58,
            f'  {"Player":<28} {"Team":>6}  {"MeanInf":>9}  {"MaxSpd":>8}',
            '─' * 58,
        ]
        for _, row in agg.iterrows():
            lines.append(
                f'  {row["name"]:<28} {row["team"]:>6}  '
                f'{row["mean_inf"]:>9.4f}  {row["max_speed"]:>7.2f}m/s')
        lines.append('═' * 58)

        report = '\n'.join(lines)
        print(report)

        txt_path = os.path.join(run_dir, 'summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as fh:
            fh.write(report)
        print(f'[OUT] Text report → {txt_path}')