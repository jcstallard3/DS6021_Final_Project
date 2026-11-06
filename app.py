"""Basic dashboard summarizing the movie datasets.

Run from the project folder with:  streamlit run app.py
Needs the cleaned files from notebooks/movie_data_wrangling.ipynb and the TMDB pull in data/.

AI disclosure: generated with Claude Code and checked by running the app against the cleaned data.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIRECTORY = Path(__file__).parent / "data"
MIN_VOTES = 1_000

# IMDb-style look on white: bright yellow for page accents, darker gold and blue for chart marks
# (bright yellow is too faint for thin lines on white)
ACCENT_YELLOW = "#F5C518"
GOLD = "#b88a00"
BLUE = "#0e63be"
INK = "#121212"
MUTED = "#5c5c5c"
FONT = '"Source Sans", "Source Sans Pro", Arial, sans-serif'  # Streamlit's page font

st.set_page_config(page_title="Are older movies better?", layout="wide")

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.5rem; max-width: 1250px; }}
.page-header {{ border-bottom: 1px solid #e5e5e5; padding-bottom: 1rem; margin-bottom: 1.5rem; }}
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
[data-testid="stTab"] p {{ color: {MUTED}; font-size: 1.05rem; font-weight: 600; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {INK}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_imdb():
    imdb = pd.read_csv(
        DATA_DIRECTORY / "processed" / "imdb_cleaned.csv",
        usecols=["tconst", "primaryTitle", "startYear", "runtimeMinutes", "genres", "averageRating",
                 "numVotes"],
    )
    imdb["decade"] = imdb["startYear"] // 10 * 10
    return imdb[imdb["decade"].between(1910, 2020)]


@st.cache_data
def load_oscars():
    oscars = pd.read_csv(DATA_DIRECTORY / "processed" / "oscars_cleaned.csv")
    oscars["decade"] = oscars["Year"] // 10 * 10
    return oscars


@st.cache_data
def oscar_films(oscars):
    """One row per film with its nomination and win counts, leaving out technical and honorary awards."""
    films = oscars[~oscars["Class"].isin(["SciTech", "Special"])].dropna(subset=["Film", "FilmId"])
    # A nomination can list several films (tt0019217|tt0018253), so split to one row per film
    films = films.assign(Film=films["Film"].str.split("|"), FilmId=films["FilmId"].str.split("|"))
    films = films.explode(["Film", "FilmId"])
    return (
        films.groupby("FilmId")
        .agg(Film=("Film", "first"), Year=("Year", "first"), Nominations=("Winner", "size"),
             Wins=("Winner", "sum"))
        .sort_values(["Nominations", "Wins"], ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data
def load_tmdb():
    tmdb = pd.read_csv(DATA_DIRECTORY / "tmdb_top10_per_year.csv")
    tmdb["year"] = pd.to_datetime(tmdb["release_date"], errors="coerce").dt.year
    tmdb["decade"] = tmdb["year"] // 10 * 10
    # TMDB records an unknown budget or revenue as 0
    tmdb[["budget", "revenue"]] = tmdb[["budget", "revenue"]].where(tmdb[["budget", "revenue"]] > 0)
    return tmdb[tmdb["decade"].between(1920, 2020)]


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
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", linecolor="#cfcfcf", tickcolor="#cfcfcf")
    return fig


def decade_line(series, title, y_title):
    fig = px.line(series.reset_index(), x=series.index.name, y=series.name, markers=True, title=title)
    fig.update_traces(line_color=GOLD, line_width=2.5, marker_size=8)
    return style(fig, "Decade", y_title)


def decade_bar(series, title, y_title):
    fig = px.bar(series.reset_index(), x=series.index.name, y=series.name, title=title)
    fig.update_traces(marker_color=GOLD)
    return style(fig, "Decade", y_title)


imdb = load_imdb()
oscars = load_oscars()
tmdb = load_tmdb()
# Few movies before the 1920s have enough votes to compare
rated = imdb[(imdb["numVotes"] >= MIN_VOTES) & (imdb["decade"] >= 1920)]

st.markdown("""
<div class="page-header">
  <h1>Are older movies better than newer ones?</h1>
  <p>A first look at three datasets: IMDb, the Academy Awards, and TMDB.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("IMDb movies since 1910", f"{len(imdb):,}")
col2.metric(f"With {MIN_VOTES:,}+ ratings", f"{len(rated):,}")
col3.metric("Oscar nominations", f"{len(oscars):,}")
col4.metric("TMDB films since 1920", f"{len(tmdb):,}")

ratings_tab, volume_tab, oscars_tab, tmdb_tab = st.tabs(
    ["Ratings", "Movies and runtime", "Oscars", "TMDB"]
)

with ratings_tab:
    section(
        "Do older movies rate higher?",
        "Across all rated movies, ratings fall in newer decades. But each decade's most-watched "
        "movies rate about the same, so older decades look better partly because mostly their "
        "best-remembered movies are still being rated.",
    )
    top_voted = rated.sort_values("numVotes", ascending=False).groupby("decade").head(100)
    comparison = pd.DataFrame({
        f"All movies with {MIN_VOTES:,}+ votes": rated.groupby("decade")["averageRating"].median(),
        "Each decade's 100 most-voted": top_voted.groupby("decade")["averageRating"].median(),
    }).reset_index().melt(id_vars="decade", var_name="Group", value_name="Median rating")

    fig = px.line(comparison, x="decade", y="Median rating", color="Group", markers=True,
                  color_discrete_sequence=[GOLD, BLUE], title="Median IMDb rating by decade")
    fig.update_traces(line_width=2.5, marker_size=8)
    fig.update_layout(legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(style(fig, "Decade", "IMDb rating"), width="stretch")

    section("How spread out are the ratings?")
    fig = px.box(rated, x="decade", y="averageRating", points=False,
                 title=f"IMDb ratings by decade (movies with {MIN_VOTES:,}+ votes)")
    fig.update_traces(marker_color=GOLD, fillcolor="#fdf3cc", line_color=GOLD)
    st.plotly_chart(style(fig, "Decade", "IMDb rating"), width="stretch")

    section("Most-voted movies by decade", "The movies people still watch and rate most from each decade.")
    decades = sorted(rated["decade"].unique().astype(int), reverse=True)
    chosen = st.selectbox("Decade", decades, index=decades.index(1950), format_func=lambda d: f"{d}s")
    decade_movies = (
        rated[rated["decade"] == chosen]
        .nlargest(15, "numVotes")
        .assign(genres=lambda df: df["genres"].str.split(","))
        [["primaryTitle", "startYear", "averageRating", "numVotes", "genres"]]
    )
    st.dataframe(
        decade_movies,
        hide_index=True,
        width="stretch",
        height=(len(decade_movies) + 1) * 35 + 3,  # tall enough to show every row
        column_config={
            "primaryTitle": st.column_config.TextColumn("Title", width="large"),
            "startYear": st.column_config.NumberColumn("Year", format="%d"),
            "averageRating": st.column_config.ProgressColumn("IMDb rating", min_value=0, max_value=10,
                                                             format="%.1f"),
            "numVotes": st.column_config.NumberColumn("Votes", format="localized"),
            "genres": st.column_config.ListColumn("Genres"),
        },
    )

with volume_tab:
    section(
        "How many movies, and how long?",
        "The 2020s are only partly over. IMDb may also list recent movies more completely.",
    )
    left, right = st.columns(2)
    movies_per_decade = imdb.groupby("decade").size().rename("Movies")
    left.plotly_chart(decade_bar(movies_per_decade, "IMDb movies released per decade", "Movies"),
                      width="stretch")
    runtime = imdb.groupby("decade")["runtimeMinutes"].median().rename("Median runtime")
    right.plotly_chart(decade_line(runtime, "Median movie runtime by decade", "Minutes"),
                       width="stretch")

with oscars_tab:
    section(
        "Academy Award nominations",
        "From about 1940 to 1966, cinematography, art direction, and costume had separate "
        "black-and-white and color awards, which doubled the nomination slots in those decades.",
    )
    left, right = st.columns(2)
    nominations = oscars.groupby("Year").size().rename("Nominations")
    fig = px.line(nominations.reset_index(), x="Year", y="Nominations",
                  title="Oscar nominations per awards year")
    fig.update_traces(line_color=GOLD, line_width=2.5)
    left.plotly_chart(style(fig, "Awards year", "Nominations"), width="stretch")

    visual = oscars[oscars["CanonicalCategory"].str.contains(
        "CINEMATOGRAPHY|ART DIRECTION|COSTUME|VISUAL EFFECTS|MAKEUP")]
    visual_films = visual.assign(FilmId=visual["FilmId"].str.split("|")).explode("FilmId")
    films_per_decade = visual_films.groupby("decade")["FilmId"].nunique().rename("Films")
    right.plotly_chart(decade_bar(films_per_decade, "Films nominated in a visual category", "Films"),
                       width="stretch")

    section("Most-nominated films", "Technical and honorary awards are left out.")
    top_films = oscar_films(oscars).head(25)
    most_nominations = int(top_films["Nominations"].max())
    st.dataframe(
        top_films,
        hide_index=True,
        width="stretch",
        column_config={
            "Film": st.column_config.TextColumn("Film", width="large"),
            "Year": st.column_config.NumberColumn("Year", format="%d"),
            "Nominations": st.column_config.ProgressColumn("Nominations", min_value=0,
                                                           max_value=most_nominations,
                                                           format="%d"),
            "Wins": st.column_config.ProgressColumn("Wins", min_value=0, max_value=most_nominations,
                                                    format="%d"),
        },
    )

with tmdb_tab:
    section(
        "Money and franchises",
        "The 10 most-voted films on TMDB for each year. Budgets are not adjusted for inflation, "
        "and films with an unknown budget are left out.",
    )
    left, right = st.columns(2)
    budget = (tmdb.groupby("decade")["budget"].median() / 1e6).rename("Median budget")
    left.plotly_chart(decade_bar(budget, "Median budget by decade (millions of dollars)", "$ millions"),
                      width="stretch")
    franchise = (tmdb.groupby("decade")["collection"].apply(lambda c: c.notna().mean()) * 100).rename(
        "Franchise share")
    right.plotly_chart(decade_bar(franchise, "Share of films in a franchise or series", "Percent"),
                       width="stretch")

    section(
        "Crews and ratings",
        "Older films may have fewer crew members listed on TMDB, not only smaller crews.",
    )
    left, right = st.columns(2)
    crew = tmdb.groupby("decade")["crew_count"].median().rename("Median crew")
    left.plotly_chart(decade_line(crew, "Median credited crew size", "Crew members"),
                      width="stretch")
    tmdb_rating = tmdb.groupby("decade")["vote_average"].median().rename("Median rating")
    right.plotly_chart(decade_line(tmdb_rating, "Median TMDB rating", "TMDB rating"),
                       width="stretch")

    section("Browse the TMDB films", "Search by title. Blank budgets and revenues are unknown.")
    search = st.text_input("Search titles", placeholder="e.g. Star Wars")
    browse = tmdb.sort_values("year", ascending=False)
    if search:
        browse = browse[browse["title"].str.contains(search, case=False, na=False, regex=False)]
    st.dataframe(
        browse.assign(genres=browse["genres"].str.split("|"), budget=browse["budget"] / 1e6,
                      revenue=browse["revenue"] / 1e6)[
            ["title", "year", "vote_average", "budget", "revenue", "collection", "crew_count", "genres"]
        ],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "title": st.column_config.TextColumn("Title", width="medium"),
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "vote_average": st.column_config.ProgressColumn("TMDB rating", min_value=0, max_value=10,
                                                            format="%.1f"),
            "budget": st.column_config.NumberColumn("Budget", format="$%.1fM"),
            "revenue": st.column_config.NumberColumn("Revenue", format="$%.1fM"),
            "collection": st.column_config.TextColumn("Franchise"),
            "crew_count": st.column_config.NumberColumn("Crew"),
            "genres": st.column_config.ListColumn("Genres"),
        },
    )
