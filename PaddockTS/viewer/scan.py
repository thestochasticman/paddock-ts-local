"""Discover and categorise the plot / video / report files a PaddockTS run
writes into ``query.out_dir``.

The pipeline writes everything for a query into one flat directory
(``{config.out_dir}/{stub}/``) with predictable name suffixes. This module
globs that directory and sorts the files into typed groups so a UI (the
Streamlit viewer, a notebook, the PDF builder) can present them without
re-deriving the naming conventions.

Categorisation keys off filename *suffixes* only — it never depends on
stub-prefixed stems — so it is robust to the SAM (``sam_paddocks_…``) vs
user (``{paddocks_stem}_…``) stem difference that the plotting stages
produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

# (suffix, human label) for the four video kinds. Most-specific suffix first
# so '_sentinel2_paddocks.mp4' matches before the shorter '_sentinel2.mp4'.
_VIDEO_KINDS = [
    ('_sentinel2_paddocks.mp4', 'Sentinel-2 + paddocks'),
    ('_fractional_cover_paddocks.mp4', 'Fractional cover + paddocks'),
    ('_sentinel2.mp4', 'Sentinel-2 true colour'),
    ('_fractional_cover.mp4', 'Fractional cover'),
]

_CALENDAR_RE = re.compile(
    r'^(?P<stem>.+)_calendar_(?P<year>\d{4})_p(?P<page>\d+)\.png$')
_PHENOLOGY_RE = re.compile(r'^(?P<stem>.+)_phenology_p(?P<page>\d+)\.png$')
_SILO_RE = re.compile(r'^(?P<stub>.+)_silo_(?P<name>.+)\.png$')
_OZWALD_RE = re.compile(
    r'^(?P<stub>.+)_ozwald_(?:daily|8day)_(?P<name>.+)\.png$')


def _variant(stem: str) -> str:
    """Map a file stem to a paddock-set label: ``'SAM'`` or the user stem."""
    return 'SAM' if stem.startswith('sam_paddocks') else stem


def _natural_key(s: str):
    """Sort key that orders embedded numbers naturally (p2 before p10)."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', s)]


@dataclass
class Asset:
    """One renderable output file plus the metadata a UI needs to group it."""

    path: Path
    label: str
    category: str            # video | climate | landscape | calendar | phenology | report
    group: str = ''          # e.g. 'SILO', 'OzWALD', 'Terrain'
    variant: str = ''        # 'SAM' or the user paddock-set stem
    year: int | None = None
    page: int | None = None

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class OutputSet:
    """All discovered outputs for one query, grouped by category."""

    out_dir: Path
    stub: str
    assets: list[Asset] = field(default_factory=list)

    def by_category(self, category: str) -> list[Asset]:
        return [a for a in self.assets if a.category == category]

    @property
    def videos(self) -> list[Asset]:
        return self.by_category('video')

    @property
    def climate(self) -> list[Asset]:
        return self.by_category('climate')

    @property
    def landscape(self) -> list[Asset]:
        return self.by_category('landscape')

    @property
    def calendars(self) -> list[Asset]:
        return self.by_category('calendar')

    @property
    def phenology(self) -> list[Asset]:
        return self.by_category('phenology')

    @property
    def report(self) -> Asset | None:
        reports = self.by_category('report')
        return reports[0] if reports else None

    @property
    def variants(self) -> list[str]:
        """Distinct paddock-set labels seen across all assets, in file order."""
        seen: list[str] = []
        for a in self.assets:
            if a.variant and a.variant not in seen:
                seen.append(a.variant)
        return seen

    def is_empty(self) -> bool:
        return not self.assets


def _classify(path: Path) -> Asset | None:
    """Return an :class:`Asset` for ``path``, or ``None`` if it isn't an output."""
    name = path.name
    suffix = path.suffix.lower()

    if suffix == '.pdf':
        label = 'PDF report' if name.endswith('_report.pdf') else name
        return Asset(path, label, 'report')

    if suffix == '.mp4':
        for sfx, label in _VIDEO_KINDS:
            if name.endswith(sfx):
                variant = _variant(name[:-len(sfx)]) if 'paddocks' in sfx else ''
                lab = f'{label} - {variant}' if variant else label
                return Asset(path, lab, 'video', variant=variant)
        return Asset(path, name, 'video')

    if suffix == '.png':
        if name.endswith('_topography.png'):
            return Asset(path, 'Topography', 'landscape', group='Terrain')

        m = _CALENDAR_RE.match(name)
        if m:
            return Asset(path, f'Calendar {m["year"]} - p{int(m["page"]):02d}',
                         'calendar', variant=_variant(m['stem']),
                         year=int(m['year']), page=int(m['page']))

        m = _PHENOLOGY_RE.match(name)
        if m:
            return Asset(path, f'Phenology - p{int(m["page"]):02d}',
                         'phenology', variant=_variant(m['stem']),
                         page=int(m['page']))

        m = _SILO_RE.match(name)
        if m:
            return Asset(path, m['name'].replace('_', ' ').title(),
                         'climate', group='SILO')

        m = _OZWALD_RE.match(name)
        if m:
            return Asset(path, m['name'].replace('_', ' ').title(),
                         'climate', group='OzWALD')

        # Unknown PNG — surface it under 'Other' rather than silently drop it.
        return Asset(path, name, 'landscape', group='Other')

    return None


def _resolve_target(target):
    """Accept a Query (duck-typed) or a directory path; return (out_dir, stub)."""
    out_dir = getattr(target, 'out_dir', None)
    stub = getattr(target, 'stub', None)
    if out_dir is not None:
        return Path(out_dir), stub or Path(out_dir).name
    p = Path(target)
    return p, p.name


def scan_outputs(target) -> OutputSet:
    """Scan a query's output directory and return a categorised OutputSet.

    Args:
        target: a :class:`PaddockTS.query.Query` (uses ``query.out_dir`` and
            ``query.stub``) or a path-like pointing directly at an output
            directory.

    Returns:
        OutputSet: every recognised file grouped by category. Empty (but
        valid) if the directory is missing or holds nothing recognisable.
    """
    out_dir, stub = _resolve_target(target)
    oset = OutputSet(out_dir=out_dir, stub=stub)
    if not out_dir.is_dir():
        return oset
    for path in sorted(out_dir.iterdir(), key=lambda p: _natural_key(p.name)):
        if not path.is_file():
            continue
        asset = _classify(path)
        if asset is not None:
            oset.assets.append(asset)
    return oset
