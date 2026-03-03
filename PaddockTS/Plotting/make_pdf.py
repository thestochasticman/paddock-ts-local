import glob
import os
from os.path import exists

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from PaddockTS.query import Query


# Ordered sections: (section_title, file_patterns)
# Each section gets a header page followed by its plots.
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
    ('Satellite Calendar', [
        ('Calendar *', '{stub}_calendar_*.png'),
    ]),
    ('Phenology', [
        ('Phenology', '{stub}_phenology.png'),
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


def make_pdf(query: Query):
    out_dir = query.out_dir
    pdf_path = f'{out_dir}/{query.stub}_report.pdf'
    os.makedirs(out_dir, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # Title page
        _add_section_page(
            pdf,
            'PaddockTS Report',
            f'{query.stub}\n{query.start} → {query.end}\n'
            f'bbox: {query.bbox}',
        )

        for section_title, plots in SECTIONS:
            # Collect all files for this section
            section_files = []
            for label, pattern in plots:
                pat = pattern.replace('{stub}', query.stub)
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
