import glob
import os
import re
from os.path import exists
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from PaddockTS.query import Query


_A4_LONG = 11.69
_A4_SHORT = 8.27
_PAGE_MARGIN = 0.4  # inches on each side
_PDF_DPI = 220       # rasterization DPI for image pages; matplotlib defaults
                     # to 100 which downsamples the embedded PNG ~3× and blurs
                     # small text. 220 keeps small labels crisp at a modest
                     # ~2-3× file-size cost vs the default.


def _natural_sort_key(s):
    """Sort strings with embedded numbers in natural order."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# Ordered sections. Each entry is (section_title, file_patterns, requires_user_paddocks).
# Patterns use {stub} for query.stub, {sam_stem} for SAM paddocks (always
# present), and {user_stem} for user paddocks (present only when the caller
# passes paddocks_filepath). Sections with requires_user_paddocks=True are
# skipped when no user file was supplied — replaces the older brittle
# `'User' in title` substring check.
SECTIONS = [
    ('Landscape', [
        ('Topography', '{stub}_topography.png'),
    ], False),
    ('Climate – SILO', [
        ('Temperature', '{stub}_silo_temperature.png'),
        ('Rainfall', '{stub}_silo_rainfall.png'),
        ('Radiation', '{stub}_silo_radiation.png'),
        ('Evapotranspiration', '{stub}_silo_evapotranspiration.png'),
        ('Humidity', '{stub}_silo_humidity.png'),
    ], False),
    ('Climate – OzWALD', [
        ('Temperature', '{stub}_ozwald_daily_temperature.png'),
        ('Precipitation', '{stub}_ozwald_daily_precipitation.png'),
        ('Wind', '{stub}_ozwald_daily_wind.png'),
        ('Radiation', '{stub}_ozwald_daily_radiation.png'),
    ], False),
    ('Satellite Calendar (SAM)', [
        ('Calendar *', '{sam_stem}_calendar_*.png'),
    ], False),
    ('Satellite Calendar (User)', [
        ('Calendar *', '{user_stem}_calendar_*.png'),
    ], True),
    ('Phenology (SAM)', [
        ('Phenology', '{sam_stem}_phenology*.png'),
    ], False),
    ('Phenology (User)', [
        ('Phenology', '{user_stem}_phenology*.png'),
    ], True),
]


def _add_section_page(pdf, title, subtitle=None):
    """Render a centred A4-landscape title page."""
    fig = plt.figure(figsize=(_A4_LONG, _A4_SHORT))
    fig.text(0.5, 0.55, title, ha='center', va='center',
             fontsize=36, fontweight='bold', color='#2c3e50')
    if subtitle:
        fig.text(0.5, 0.42, subtitle, ha='center', va='center',
                 fontsize=16, color='#7f8c8d')
    fig.patch.set_facecolor('white')
    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf, image_path):
    """Lay out ``image_path`` centred on an A4-landscape page."""
    img = Image.open(image_path)
    w, h = img.size

    page_w, page_h = _A4_LONG, _A4_SHORT  # landscape A4
    max_w = page_w - 2 * _PAGE_MARGIN
    max_h = page_h - 2 * _PAGE_MARGIN

    # Fit the image into the printable area, preserving aspect. The /100
    # factor converts pixel dims to "natural inches" assuming a 100-DPI
    # source — same convention as before.
    scale = min(max_w / (w / 100), max_h / (h / 100))
    fig_w = w / 100 * scale
    fig_h = h / 100 * scale

    fig = plt.figure(figsize=(page_w, page_h))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([
        (page_w - fig_w) / 2 / page_w,
        (page_h - fig_h) / 2 / page_h,
        fig_w / page_w,
        fig_h / page_h,
    ])
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig, dpi=_PDF_DPI)
    plt.close(fig)
    img.close()


def make_pdf(query: Query, paddocks_filepath: str | None = None,
             label_col: str | None = None):
    """Generate a PDF report combining all plots for a query.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        paddocks_filepath: Optional path to the user-provided paddocks file.
            If provided, includes user paddock calendar and phenology plots
            in the report.
        label_col: Column in the user paddocks file to use for per-row
            labels in the user-paddocks calendar pages. Ignored for the
            SAM calendar (which always uses the numeric ``paddock`` id).

    Returns:
        str: Filesystem path of the generated PDF.
    """
    out_dir = query.out_dir
    pdf_path = f'{out_dir}/{query.stub}_report.pdf'
    os.makedirs(out_dir, exist_ok=True)

    # Derive stems for SAM and user paddocks
    sam_stem = f'{query.stub}_sam_paddocks'
    if paddocks_filepath is not None:
        user_stem = Path(paddocks_filepath).stem
    else:
        user_stem = ''  # Will result in no matches for user sections

    with PdfPages(pdf_path) as pdf:
        # PDF document metadata — shows up in viewer tabs, file properties,
        # and search indexes. Cheap to set; helpful for users juggling many
        # reports across queries.
        meta = pdf.infodict()
        meta['Title'] = f'PaddockTS Report — {query.stub}'
        meta['Author'] = 'PaddockTS'
        meta['Subject'] = (
            f'Paddock-scale time-series analysis for bbox {query.bbox} '
            f'between {query.start} and {query.end}'
        )
        meta['Keywords'] = 'PaddockTS, Sentinel-2, phenology, fractional cover'
        meta['Creator'] = 'PaddockTS.Plotting.make_pdf'

        # Cover page (always landscape — it's centred text, orientation
        # doesn't affect legibility and the cover is the user's first
        # impression).
        _add_section_page(
            pdf,
            'PaddockTS Report',
            f'{query.stub}\n{query.start} → {query.end}\n'
            f'bbox: {query.bbox}',
        )

        for section_title, plots, requires_user in SECTIONS:
            if requires_user and paddocks_filepath is None:
                continue

            # Calendar sections are special: instead of embedding the cached
            # PNGs (which gives raster text that shrinks badly in the embed),
            # we re-render the matplotlib Figures and write them directly
            # to the PDF so the labels stay vector text.
            if 'Calendar' in section_title:
                from PaddockTS.Plotting.calendar_plot import iter_calendar_figures
                cal_paddocks = (paddocks_filepath if requires_user
                                else query.sam_paddocks_path)
                if not exists(cal_paddocks):
                    continue
                cal_label_col = label_col if requires_user else None
                _add_section_page(pdf, section_title)
                for _year, _page_idx, fig in iter_calendar_figures(
                    query, paddocks_filepath=cal_paddocks,
                    label_col=cal_label_col,
                ):
                    pdf.savefig(fig)
                    plt.close(fig)
                continue

            # Collect all files for this section
            section_files = []
            for label, pattern in plots:
                pat = pattern.replace('{stub}', query.stub)
                pat = pat.replace('{sam_stem}', sam_stem)
                pat = pat.replace('{user_stem}', user_stem)
                matches = sorted(glob.glob(f'{out_dir}/{pat}'), key=_natural_sort_key)
                for m in matches:
                    section_files.append(m)

            if not section_files:
                continue

            _add_section_page(pdf, section_title)

            for path in section_files:
                _add_image_page(pdf, path)

    print(f'PDF report saved to {pdf_path}')
    return pdf_path


if __name__ == '__main__':
    # Build a PDF for the repo-bundled artifacts. Assumes the upstream
    # pipeline has already produced calendar / phenology / topography PNGs
    # for this query under query.out_dir; if not, sections silently skip.
    from datetime import date
    fp = 'artifacts/PaddockSet1.gpkg'
    query = Query.build_from_paddocks(fp, date(2024, 1, 1), date(2025, 1, 1), 'PaddockSet1')
    make_pdf(query, paddocks_filepath=fp)