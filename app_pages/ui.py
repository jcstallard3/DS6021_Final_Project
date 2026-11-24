"""Shared look for every page: colors, page and section headings, chart styling, and data checks."""
from pathlib import Path

import streamlit as st

PROJECT_DIRECTORY = Path(__file__).parent.parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
RAW_DIRECTORY = DATA_DIRECTORY / "raw"
PROCESSED_DIRECTORY = DATA_DIRECTORY / "processed"

# IMDb-style look on white: bright yellow for page accents, darker gold and blue for chart marks
# (bright yellow is too faint for thin lines on white)
ACCENT_YELLOW = "#F5C518"
GOLD = "#b88a00"
BLUE = "#0e63be"
INK = "#121212"
MUTED = "#5c5c5c"
GRID = "#eeeeee"
FONT = '"Source Sans", "Source Sans Pro", Arial, sans-serif'  # Streamlit's page font

CSS = f"""
<style>
.block-container {{ padding-top: 2.5rem; max-width: 1250px; }}
.page-header {{ border-bottom: 1px solid #e5e5e5; padding-bottom: 1rem; margin-bottom: 1.2rem; }}
.page-header h1 {{ color: {INK}; font-weight: 800; font-size: 2.3rem; margin: 0; padding: 0; }}
.page-header p {{ color: {MUTED}; font-size: 1.05rem; margin: 0.35rem 0 0; }}
.section-title {{
    border-left: 4px solid {ACCENT_YELLOW}; padding-left: 0.65rem; margin: 1.6rem 0 0.2rem;
    color: {INK}; font-weight: 700; font-size: 1.4rem;
}}
.section-note {{ color: {MUTED}; margin: 0 0 0.6rem 0.9rem; }}
div[data-testid="stMetric"] {{
    background: #ffffff; border: 1px solid #e5e5e5; border-top: 4px solid {ACCENT_YELLOW};
    border-radius: 6px; padding: 0.9rem 1.1rem; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
}}
div[data-testid="stMetricValue"] {{ color: {INK}; font-weight: 800; }}
div[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 600; }}
/* Selected section button: yellow fill with dark text (yellow text on white is unreadable) */
[data-testid="stButtonGroup"] button[aria-checked="true"] {{
    background: {ACCENT_YELLOW}; border-color: {ACCENT_YELLOW};
}}
[data-testid="stButtonGroup"] button[aria-checked="true"] p {{ color: {INK}; font-weight: 700; }}
</style>
"""


def page_header(title, subtitle):
    st.markdown(f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def subpages(options, key):
    """The row of section buttons under a page header; returns the selected section."""
    return st.segmented_control("Section", options, default=options[0], required=True, key=key,
                                label_visibility="collapsed")


def section(title, note=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<p class="section-note">{note}</p>', unsafe_allow_html=True)


def style(fig, x_title, y_title):
    fig.update_layout(
        template="simple_white",
        font={"family": FONT, "color": MUTED, "size": 13},
        title_font={"family": FONT, "color": INK, "size": 16},
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="",
        hoverlabel={"bgcolor": "white", "font_family": FONT},
        margin={"t": 50, "l": 10, "r": 10},
    )
    fig.update_xaxes(linecolor="#cfcfcf", tickcolor="#cfcfcf")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor="#cfcfcf", tickcolor="#cfcfcf")
    return fig


def table(data, **kwargs):
    st.dataframe(data, hide_index=True, width="stretch", **kwargs)


def missing_files(paths, how_to_get):
    """Show which data files are missing. Returns True when the page can't run."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        listed = "\n".join(f"- `{path.relative_to(PROJECT_DIRECTORY).as_posix()}`" for path in missing)
        st.warning(f"This page needs data files that aren't on this computer yet:\n\n{listed}\n\n"
                   f"{how_to_get}")
    return bool(missing)
