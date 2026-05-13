import glob
import os
from os.path import exists
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from PaddockTS.query import Query


# Ordered sections: (section_title, file_patterns)
# Each section gets a header page followed by its plots.
# Patterns use {sam_stem} for SAM paddocks and {user_stem} for user paddocks
SECTIONS = [
    ('Landscape', [
        ('Topography', '{stub}_topography.png'),
    ]),
    ('Climate – SILO', [
        ('Temperature', '{stub}_silo_temperature.png'),
        ('Rainfall', '{stub}_silo_rainfall.png'),
        ('Radiation', '{stub}_silo_radiation.png'),
        ('Evapotranspiration', '{stub}_silo_evapotranspiration.png'),
        ('Humidity', '{stub}_silo_humidity.png'),
    ]),
    ('Climate – OzWALD', [
        ('Temperature', '{stub}_ozwald_daily_temperature.png'),
        ('Precipitation', '{stub}_ozwald_daily_precipitation.png'),
        ('Wind', '{stub}_ozwald_daily_wind.png'),
        ('Radiation', '{stub}_ozwald_daily_radiation.png'),
    ]),
    ('Satellite Calendar (SAM)', [
        ('Calendar *', '{sam_stem}_calendar_*.png'),
    ]),
    ('Satellite Calendar (User)', [
        ('Calendar *', '{user_stem}_calendar_*.png'),
    ]),
    ('Phenology (SAM)', [
        ('Phenology', '{sam_stem}_phenology.png'),
    ]),
    ('Phenology (User)', [
        ('Phenology', '{user_stem}_phenology.png'),
    ]),
]


def _add_section_page(pdf, title, subtitle=None):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.text(0.5, 0.55, title, ha='center', va='center',
             fontsize=36, fontweight='bold', color='#2c3e50')
    if subtitle:
        fig.text(0.5, 0.42, subtitle, ha='center', va='center',
                 fontsize=16, color='#7f8c8d')
    fig.patch.set_facecolor('white')
    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf, image_path):
    img = Image.open(image_path)
    w, h = img.size
    # Fit into A4 landscape with margins
    max_w, max_h = 11.0, 7.5
    scale = min(max_w / (w / 100), max_h / (h / 100))
    fig_w = w / 100 * scale
    fig_h = h / 100 * scale

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([
        (11.69 - fig_w) / 2 / 11.69,
        (8.27 - fig_h) / 2 / 8.27,
        fig_w / 11.69,
        fig_h / 8.27,
    ])
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig)
    plt.close(fig)
    img.close()


def make_pdf(query: Query, paddocks_filepath: str | None = None):
    """Generate a PDF report combining all plots for a query.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        paddocks_filepath: Optional path to the user-provided paddocks file.
            If provided, includes user paddock calendar and phenology plots
            in the report.

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
        # Title page
        _add_section_page(
            pdf,
            'PaddockTS Report',
            f'{query.stub}\n{query.start} → {query.end}\n'
            f'bbox: {query.bbox}',
        )

        for section_title, plots in SECTIONS:
            # Skip user sections if no user paddocks provided
            if 'User' in section_title and paddocks_filepath is None:
                continue

            # Collect all files for this section
            section_files = []
            for label, pattern in plots:
                pat = pattern.replace('{stub}', query.stub)
                pat = pat.replace('{sam_stem}', sam_stem)
                pat = pat.replace('{user_stem}', user_stem)
                matches = sorted(glob.glob(f'{out_dir}/{pat}'))
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
    from PaddockTS.utils import get_example_query
    make_pdf(get_example_query())
