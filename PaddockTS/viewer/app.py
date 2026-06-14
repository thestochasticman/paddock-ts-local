"""Streamlit viewer for PaddockTS outputs.

Don't run this file by hand — launch it through the helper, which handles
local-vs-SSH and the port-forward hint::

    from PaddockTS.viewer import serve
    serve(query)                 # local: opens your browser
    serve(query, headless=True)  # SSH: bind a port, you forward it

Under the hood the launcher runs::

    streamlit run .../viewer/app.py -- --out-dir <dir> --stub <stub>
"""
import argparse
import base64

import streamlit as st
import streamlit.components.v1 as components

from PaddockTS.viewer.scan import scan_outputs


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--stub', default=None)
    # Streamlit forwards script args after `--`; tolerate anything extra.
    args, _ = parser.parse_known_args()
    return args


def _video_tab(oset):
    vids = oset.videos
    if not vids:
        st.info('No videos found for this run.')
        return
    cols = st.columns(2)
    for i, a in enumerate(vids):
        with cols[i % 2]:
            st.caption(a.label)
            st.video(str(a.path))


def _calendar_tab(oset):
    cals = oset.calendars
    if not cals:
        st.info('No calendar plots found.')
        return
    variants = sorted({a.variant for a in cals})
    variant = st.selectbox('Paddock set', variants, key='cal_variant')
    sub = [a for a in cals if a.variant == variant]
    years = sorted({a.year for a in sub})
    year = st.selectbox('Year', years, key='cal_year')
    pages = sorted([a for a in sub if a.year == year],
                   key=lambda a: a.page or 0)
    for a in pages:
        st.image(str(a.path), caption=a.label, use_container_width=True)


def _phenology_tab(oset):
    phen = oset.phenology
    if not phen:
        st.info('No phenology plots found.')
        return
    variants = sorted({a.variant for a in phen})
    variant = st.selectbox('Paddock set', variants, key='phen_variant')
    pages = sorted([a for a in phen if a.variant == variant],
                   key=lambda a: a.page or 0)
    for a in pages:
        st.image(str(a.path), caption=a.label, use_container_width=True)


def _climate_tab(oset):
    clim = oset.climate
    if not clim:
        st.info('No climate plots found.')
        return
    for group in ('SILO', 'OzWALD'):
        items = [a for a in clim if a.group == group]
        if not items:
            continue
        st.subheader(f'Climate - {group}')
        cols = st.columns(2)
        for i, a in enumerate(items):
            with cols[i % 2]:
                st.image(str(a.path), caption=a.label,
                         use_container_width=True)


def _landscape_tab(oset):
    land = oset.landscape
    if not land:
        st.info('No landscape / terrain plots found.')
        return
    for a in land:
        st.image(str(a.path), caption=a.label, use_container_width=True)


def _report_tab(oset):
    rep = oset.report
    if rep is None:
        st.info('No PDF report found.')
        return
    data = rep.path.read_bytes()
    st.download_button('Download PDF', data, file_name=rep.path.name,
                       mime='application/pdf')
    b64 = base64.b64encode(data).decode()
    components.html(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="900" style="border:none;"></iframe>',
        height=920,
    )


def main():
    args = _parse_args()
    st.set_page_config(page_title='PaddockTS viewer', layout='wide')

    oset = scan_outputs(args.out_dir)
    title = args.stub or oset.stub

    st.sidebar.title('PaddockTS')
    st.sidebar.markdown(f'**Run:** `{title}`')
    st.sidebar.caption(str(oset.out_dir))
    if st.sidebar.button('Rescan outputs'):
        st.rerun()

    if oset.is_empty():
        st.warning(f'No PaddockTS outputs found in `{oset.out_dir}`. '
                   'Has the pipeline finished writing this run?')
        return

    tabs = st.tabs(['Videos', 'Calendars', 'Phenology',
                    'Climate', 'Landscape', 'Report'])
    with tabs[0]:
        _video_tab(oset)
    with tabs[1]:
        _calendar_tab(oset)
    with tabs[2]:
        _phenology_tab(oset)
    with tabs[3]:
        _climate_tab(oset)
    with tabs[4]:
        _landscape_tab(oset)
    with tabs[5]:
        _report_tab(oset)


main()
