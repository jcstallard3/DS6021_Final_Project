"""Dashboard for the DS 6021 movie project: our exploratory analysis and each group member's model.

Run from the project folder with:  streamlit run app.py

Each page lives in its own file in app_pages/, and each page's sections are the row of buttons under
its header:
- eda.py: exploratory data analysis (IMDb, Oscars, TMDB)
- era.py: era comparison (notebooks/era_comparison.ipynb)
- awards.py: Oscar prediction (notebooks/award_modeling.ipynb)
- recommender.py: movie recommender (notebooks/movie_rating_knn.ipynb)

AI disclosure: generated with Claude Code and checked by running the app against the project data
(see AI_USE.md).
"""
import streamlit as st

from app_pages import awards, eda, era, recommender
from app_pages.ui import CSS

st.set_page_config(page_title="Old vs. new movies", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

page = st.navigation(
    [
        st.Page(eda.render, title="Exploratory data analysis", url_path="eda", default=True),
        st.Page(era.render, title="Era comparison", url_path="era-comparison"),
        st.Page(awards.render, title="Oscar prediction", url_path="oscar-prediction"),
        st.Page(recommender.render, title="Movie recommender", url_path="recommender"),
    ],
    position="top",
)
page.run()
