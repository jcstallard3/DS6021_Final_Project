"""Exploratory data analysis page: IMDb ratings, movie counts and runtime, the Oscars, and TMDB."""
import pandas as pd
import plotly.express as px
import streamlit as st

from app_pages.ui import (BLUE, GOLD, PROCESSED_DIRECTORY, page_header, section, style, subpages,
                          table)

MIN_VOTES = 1_000


@st.cache_data
def load_imdb():
    imdb = pd.read_csv(
        PROCESSED_DIRECTORY / "imdb_cleaned.csv",
        usecols=["tconst", "primaryTitle", "startYear", "runtimeMinutes", "genres", "averageRating",
                 "numVotes"],
    )
    imdb["decade"] = imdb["startYear"] // 10 * 10
    return imdb[imdb["decade"].between(1910, 2020)]


@st.cache_data
def load_oscars():
    oscars = pd.read_csv(PROCESSED_DIRECTORY / "oscars_cleaned.csv")
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
    tmdb = pd.read_csv(PROCESSED_DIRECTORY / "tmdb_top10_per_year.csv")
    tmdb["year"] = pd.to_datetime(tmdb["release_date"], errors="coerce").dt.year
    tmdb["decade"] = tmdb["year"] // 10 * 10
    # TMDB records an unknown budget or revenue as 0
    tmdb[["budget", "revenue"]] = tmdb[["budget", "revenue"]].where(tmdb[["budget", "revenue"]] > 0)
    return tmdb[tmdb["decade"].between(1920, 2020)]


def money(amount):
    """$90.6M under a billion, $1.15B from a billion up, blank when unknown."""
    if pd.isna(amount):
        return ""
    if amount >= 1e9:
        return f"${amount / 1e9:.2f}B"
    return f"${amount / 1e6:.1f}M"


def decade_line(series, title, y_title):
    fig = px.line(series.reset_index(), x=series.index.name, y=series.name, markers=True, title=title)
    fig.update_traces(line_color=GOLD, line_width=2.5, marker_size=8)
    return style(fig, "Decade", y_title)


def decade_bar(series, title, y_title):
    fig = px.bar(series.reset_index(), x=series.index.name, y=series.name, title=title)
    fig.update_traces(marker_color=GOLD)
    return style(fig, "Decade", y_title)


def show_ratings(rated):
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
    table(
        decade_movies,
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


def show_volume(imdb):
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


def show_oscars(oscars):
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
    table(
        top_films,
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


def show_tmdb(tmdb):
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
    browse = browse.assign(genres=browse["genres"].str.split("|"))[
        ["title", "year", "vote_average", "budget", "revenue", "collection", "crew_count", "genres"]
    ]
    table(
        # Shown as $90.6M, or $1.15B from a billion up; sorting still uses the real amount
        browse.style.format(money, subset=["budget", "revenue"]),
        height=420,
        column_config={
            "title": st.column_config.TextColumn("Title", width="medium"),
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "vote_average": st.column_config.ProgressColumn("TMDB rating", min_value=0, max_value=10,
                                                            format="%.1f"),
            "budget": st.column_config.Column("Budget"),
            "revenue": st.column_config.Column("Revenue"),
            "collection": st.column_config.TextColumn("Franchise"),
            "crew_count": st.column_config.NumberColumn("Crew"),
            "genres": st.column_config.ListColumn("Genres"),
        },
    )


def render():
    page_header("Exploratory data analysis",
                "A first look at three datasets: IMDb, the Academy Awards, and TMDB.")
    imdb = load_imdb()
    oscars = load_oscars()
    tmdb = load_tmdb()
    # Few movies before the 1920s have enough votes to compare
    rated = imdb[(imdb["numVotes"] >= MIN_VOTES) & (imdb["decade"] >= 1920)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IMDb movies since 1910", f"{len(imdb):,}")
    col2.metric(f"With {MIN_VOTES:,}+ ratings", f"{len(rated):,}")
    col3.metric("Oscar nominations", f"{len(oscars):,}")
    col4.metric("TMDB films since 1920", f"{len(tmdb):,}")

    topic = subpages(["Ratings", "Movies and runtime", "Oscars", "TMDB"], key="eda_section")
    if topic == "Ratings":
        show_ratings(rated)
    elif topic == "Movies and runtime":
        show_volume(imdb)
    elif topic == "Oscars":
        show_oscars(oscars)
    else:
        show_tmdb(tmdb)
