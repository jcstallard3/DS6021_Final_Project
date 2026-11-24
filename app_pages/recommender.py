"""Movie recommender page: item-based KNN on MovieLens rating patterns, and a ratings + genres hybrid.

Same steps and settings as notebooks/movie_rating_knn.ipynb.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from app_pages.ui import (BLUE, GOLD, PROCESSED_DIRECTORY, missing_files, page_header, section,
                          style, subpages, table)

USER_COL = "userId"
ITEM_COL = "movieId"
RATING_COL = "rating"
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE
RANDOM_STATE = 42
MIN_MOVIE_RATINGS = 5
TOP_N = 10
MIN_POSITIVE_RATING = 4.0
KNN_CANDIDATES = 100

MIN_RATINGS_OPTIONS = [3, 5, 10]
MIN_SHARED_OPTIONS = [1, 3, 5]
CANDIDATE_OPTIONS = [50, 100, 200]
RATING_WEIGHTS = [0.3, 0.5, 0.7, 0.9]

# Chosen by the tuning in the notebook
BEST_MIN_MOVIE_RATINGS = 10
BEST_MIN_SHARED_USERS = 5
BEST_CANDIDATES = 200
BEST_HYBRID_WEIGHT = 0.90

FILES = [PROCESSED_DIRECTORY / "movielens_ratings_cleaned.csv",
         PROCESSED_DIRECTORY / "movielens_movies_cleaned.csv"]


@st.cache_data(show_spinner="Loading MovieLens ratings...")
def load_data():
    ratings = pd.read_csv(PROCESSED_DIRECTORY / "movielens_ratings_cleaned.csv")
    movies = pd.read_csv(PROCESSED_DIRECTORY / "movielens_movies_cleaned.csv")
    prepared = (
        ratings[[USER_COL, ITEM_COL, RATING_COL]]
        .dropna()
        .drop_duplicates(subset=[USER_COL, ITEM_COL])
        .copy()
    )
    train_val, test = train_test_split(prepared, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                  random_state=RANDOM_STATE)
    return ratings, movies, train, val, test


def create_movie_user_matrix(ratings, min_ratings=MIN_MOVIE_RATINGS):
    """Create a sparse movie-user matrix for movies with enough ratings."""
    movie_counts = ratings.groupby(ITEM_COL).size()
    eligible_movies = movie_counts[movie_counts >= min_ratings].index
    filtered = ratings[ratings[ITEM_COL].isin(eligible_movies)]
    matrix = filtered.pivot(index=ITEM_COL, columns=USER_COL, values=RATING_COL)
    return matrix, csr_matrix(matrix.fillna(0).to_numpy())


def train_item_knn(movie_matrix):
    """Fit a KNN model using movie rating patterns."""
    return NearestNeighbors(metric="cosine", algorithm="brute").fit(movie_matrix)


def fit_popularity_baseline(ratings, min_ratings=MIN_MOVIE_RATINGS):
    """Calculate movie averages and keep movies with enough ratings."""
    movie_stats = (
        ratings.groupby(ITEM_COL)[RATING_COL]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )
    return movie_stats[movie_stats["rating_count"] >= min_ratings].sort_values(
        ["avg_rating", "rating_count"], ascending=False)


@st.cache_resource(show_spinner="Training the KNN models...")
def fitted_models():
    """Every fitted model the page uses, trained on the training split only."""
    _, movies, train, _, _ = load_data()
    matrix, sparse = create_movie_user_matrix(train)
    tuned_matrix, tuned_sparse = create_movie_user_matrix(train, BEST_MIN_MOVIE_RATINGS)
    return {
        "popular": fit_popularity_baseline(train),
        "matrix": matrix, "sparse": sparse, "knn": train_item_knn(sparse),
        "tuned_matrix": tuned_matrix, "tuned_sparse": tuned_sparse,
        "tuned_knn": train_item_knn(tuned_sparse),
        "hybrid": fit_hybrid_model(train, movies, BEST_MIN_MOVIE_RATINGS),
    }


def recommend_popular_movies(movie_stats, movies, n=TOP_N, exclude_movie_id=None):
    """Return the highest-rated movies with their titles."""
    recommendations = movie_stats
    if exclude_movie_id is not None:
        recommendations = recommendations[recommendations[ITEM_COL] != exclude_movie_id]
    return (
        recommendations.head(n)
        .merge(movies[[ITEM_COL, "title"]], on=ITEM_COL, how="left")
        [["title", "avg_rating", "rating_count"]]
    )


def get_knn_recommendations(models, movie_id, seen_movies, n=TOP_N):
    """Recommend similar unseen movies using the first (untuned) KNN model."""
    matrix = models["matrix"]
    movie_index = matrix.index.get_loc(movie_id)
    neighbor_count = min(KNN_CANDIDATES + 1, len(matrix))
    distances, indices = models["knn"].kneighbors(models["sparse"][movie_index],
                                                  n_neighbors=neighbor_count)
    neighbors = pd.DataFrame({ITEM_COL: matrix.index[indices[0]], "similarity": 1 - distances[0]})
    return neighbors[~neighbors[ITEM_COL].isin(seen_movies)].head(n)[ITEM_COL].tolist()


def evaluate_recommender(models, train, validation, method="knn", n=TOP_N):
    """Measure Recall@N and Hit Rate@N using validation ratings."""
    positive_train = train[train[RATING_COL] >= MIN_POSITIVE_RATING]
    positive_val = validation[validation[RATING_COL] >= MIN_POSITIVE_RATING]
    seen_by_user = train.groupby(USER_COL)[ITEM_COL].apply(set)
    liked_by_user = positive_val.groupby(USER_COL)[ITEM_COL].apply(set)
    popular_movies = models["popular"]

    hits = total_liked = successful_users = evaluated_users = total_recommended = 0
    for user_id, user_ratings in positive_train.groupby(USER_COL):
        if user_id not in liked_by_user:
            continue
        seed_movie = user_ratings.sort_values([RATING_COL, ITEM_COL],
                                              ascending=[False, True])[ITEM_COL].iloc[0]
        if seed_movie not in models["matrix"].index:
            continue
        seen_movies = seen_by_user[user_id]
        liked_movies = liked_by_user[user_id]
        if method == "knn":
            recommendations = get_knn_recommendations(models, seed_movie, seen_movies, n)
        else:
            recommendations = (
                popular_movies[~popular_movies[ITEM_COL].isin(seen_movies)].head(n)[ITEM_COL].tolist()
            )
        user_hits = len(set(recommendations) & liked_movies)
        hits += user_hits
        total_liked += len(liked_movies)
        successful_users += int(user_hits > 0)
        evaluated_users += 1
        total_recommended += len(recommendations)

    return {"Recall@10": hits / total_liked, "Hit Rate@10": successful_users / evaluated_users,
            "Users Evaluated": evaluated_users,
            "Avg Recommendations": total_recommended / evaluated_users}


def prepare_evaluation_cases(train, validation, eligible_movies):
    """Create fixed user histories and relevant movies for evaluation."""
    positive_train = train[train[RATING_COL] >= MIN_POSITIVE_RATING]
    positive_val = validation[validation[RATING_COL] >= MIN_POSITIVE_RATING]
    seen_by_user = train.groupby(USER_COL)[ITEM_COL].apply(set)
    liked_by_user = positive_val.groupby(USER_COL)[ITEM_COL].apply(set)
    cases = []
    for user_id, ratings in positive_train.groupby(USER_COL):
        if user_id not in liked_by_user:
            continue
        seed = ratings.sort_values([RATING_COL, ITEM_COL], ascending=[False, True])[ITEM_COL].iloc[0]
        if seed not in eligible_movies:
            continue
        cases.append({"user_id": user_id, "seed": seed, "seen": seen_by_user[user_id],
                      "liked": liked_by_user[user_id]})
    return cases


def build_candidate_cache(train, cases, min_ratings, max_candidates):
    """Train KNN and find candidate movies with their shared-user counts."""
    matrix, sparse = create_movie_user_matrix(train, min_ratings=min_ratings)
    model = train_item_knn(sparse)
    binary = sparse.copy()
    binary.data = np.ones_like(binary.data)
    valid_seeds = sorted({case["seed"] for case in cases if case["seed"] in matrix.index})
    cache = {}
    if not valid_seeds:
        return cache
    seed_indices = matrix.index.get_indexer(valid_seeds)
    neighbor_count = min(max_candidates + 1, len(matrix))
    _, indices = model.kneighbors(sparse[seed_indices], n_neighbors=neighbor_count)
    for seed, seed_index, neighbor_indices in zip(valid_seeds, seed_indices, indices):
        shared_counts = (binary[neighbor_indices] @ binary[seed_index].T).toarray().ravel()
        cache[seed] = [(matrix.index[i], int(shared))
                       for i, shared in zip(neighbor_indices, shared_counts)
                       if matrix.index[i] != seed]
    return cache


def evaluate_configuration(cases, cache, min_shared, candidates, popular_movies):
    """Evaluate recommendations with fixed user histories."""
    total_hits = total_liked = successful_users = total_recommendations = fallback_users = 0
    for case in cases:
        seen = case["seen"]
        neighbors = cache.get(case["seed"])
        if neighbors is None:
            fallback_users += 1
            recommendations = popular_movies.loc[
                ~popular_movies[ITEM_COL].isin(seen), ITEM_COL].head(TOP_N).tolist()
        else:
            recommendations = [movie_id for movie_id, shared in neighbors[:candidates]
                               if shared >= min_shared and movie_id not in seen][:TOP_N]
        hits = len(set(recommendations) & case["liked"])
        total_hits += hits
        total_liked += len(case["liked"])
        successful_users += int(hits > 0)
        total_recommendations += len(recommendations)
    n_users = len(cases)
    return {"recall": total_hits / total_liked, "hit_rate": successful_users / n_users,
            "avg_recommendations": total_recommendations / n_users,
            "fallback_users": fallback_users, "users_evaluated": n_users}


def fit_hybrid_model(ratings, movies, min_ratings=BEST_MIN_MOVIE_RATINGS):
    """Prepare normalized rating and genre matrices for movie similarity."""
    matrix, sparse = create_movie_user_matrix(ratings, min_ratings)
    genres = (
        movies.set_index(ITEM_COL)["genres"].reindex(matrix.index).fillna("").str.get_dummies(sep="|")
    )
    rating_vectors = normalize(sparse)
    genre_vectors = normalize(csr_matrix(genres.to_numpy()))
    binary = sparse.copy()
    binary.data = np.ones_like(binary.data)
    return matrix, rating_vectors, genre_vectors, binary


def find_hybrid_neighbors(movie_id, hybrid_model, weight, candidates):
    """Find similar movies using weighted rating and genre similarities."""
    matrix, rating_vectors, genre_vectors, binary = hybrid_model
    if movie_id not in matrix.index:
        return []
    movie_index = matrix.index.get_loc(movie_id)
    rating_similarity = (rating_vectors @ rating_vectors[movie_index].T).toarray().ravel()
    genre_similarity = (genre_vectors @ genre_vectors[movie_index].T).toarray().ravel()
    combined_similarity = weight * rating_similarity + (1 - weight) * genre_similarity
    combined_similarity[movie_index] = -np.inf
    neighbor_indices = np.argsort(-combined_similarity, kind="stable")[:candidates]
    shared_counts = (binary[neighbor_indices] @ binary[movie_index].T).toarray().ravel()
    return [(matrix.index[i], int(shared), combined_similarity[i])
            for i, shared in zip(neighbor_indices, shared_counts)]


def recommend_tuned_knn(models, movie_id, movies, n=TOP_N):
    """Find similar movies that meet the shared-user requirement."""
    matrix, sparse = models["tuned_matrix"], models["tuned_sparse"]
    movie_index = matrix.index.get_loc(movie_id)
    neighbor_count = min(BEST_CANDIDATES + 1, len(matrix))
    distances, indices = models["tuned_knn"].kneighbors(sparse[movie_index], n_neighbors=neighbor_count)
    binary = sparse.copy()
    binary.data = np.ones_like(binary.data)
    neighbor_indices = indices[0]
    shared_counts = (binary[neighbor_indices] @ binary[movie_index].T).toarray().ravel()
    recommendations = pd.DataFrame({ITEM_COL: matrix.index[neighbor_indices],
                                    "similarity": 1 - distances[0], "shared_users": shared_counts})
    recommendations = recommendations[
        (recommendations[ITEM_COL] != movie_id)
        & (recommendations["shared_users"] >= BEST_MIN_SHARED_USERS)
    ].head(n)
    return recommendations.merge(movies[[ITEM_COL, "title"]], on=ITEM_COL, how="left")[
        ["title", "similarity", "shared_users"]].reset_index(drop=True)


def recommend_hybrid(models, movie_id, movies, weight=BEST_HYBRID_WEIGHT, n=TOP_N):
    neighbors = find_hybrid_neighbors(movie_id, models["hybrid"], weight, BEST_CANDIDATES)
    recommendations = pd.DataFrame(neighbors, columns=[ITEM_COL, "shared_users", "similarity"])
    return (
        recommendations[recommendations["shared_users"] >= BEST_MIN_SHARED_USERS]
        .head(n)
        .merge(movies[[ITEM_COL, "title"]], on=ITEM_COL, how="left")
        [["title", "similarity", "shared_users"]]
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner="Evaluating the recommenders on validation users...")
def evaluation_results():
    _, movies, train, val, _ = load_data()
    models = fitted_models()
    baseline = evaluate_recommender(models, train, val, method="baseline")
    knn = evaluate_recommender(models, train, val, method="knn")

    cases = prepare_evaluation_cases(train, val, models["matrix"].index)
    tuning = []
    for min_ratings in MIN_RATINGS_OPTIONS:
        cache = build_candidate_cache(train, cases, min_ratings, max(CANDIDATE_OPTIONS))
        for min_shared in MIN_SHARED_OPTIONS:
            for candidates in CANDIDATE_OPTIONS:
                metrics = evaluate_configuration(cases, cache, min_shared, candidates, models["popular"])
                tuning.append({"min_movie_ratings": min_ratings, "min_shared_users": min_shared,
                               "candidates": candidates, **metrics})
    tuning = pd.DataFrame(tuning).sort_values(["hit_rate", "recall"], ascending=False)
    tuning = tuning.reset_index(drop=True)

    rating_cache = build_candidate_cache(train, cases, BEST_MIN_MOVIE_RATINGS, BEST_CANDIDATES)
    rating_results = evaluate_configuration(cases, rating_cache, BEST_MIN_SHARED_USERS,
                                            BEST_CANDIDATES, models["popular"])
    hybrid = []
    for weight in RATING_WEIGHTS:
        hybrid_cache = {}
        for case in cases:
            seed = case["seed"]
            if seed not in hybrid_cache:
                neighbors = find_hybrid_neighbors(seed, models["hybrid"], weight, BEST_CANDIDATES)
                hybrid_cache[seed] = [(movie_id, shared) for movie_id, shared, _ in neighbors]
        metrics = evaluate_configuration(cases, hybrid_cache, BEST_MIN_SHARED_USERS, BEST_CANDIDATES,
                                         models["popular"])
        hybrid.append({"rating_weight": weight, "genre_weight": 1 - weight, **metrics})
    hybrid = pd.DataFrame(hybrid)
    return {"baseline": baseline, "knn": knn, "tuning": tuning, "rating": rating_results,
            "hybrid": hybrid}


def show_overview(ratings, movies, train, val, test, models):
    section("The idea",
            "Recommend movies based on how similarly people rated them. If a user likes Toy Story, "
            "the model looks for other movies with similar rating patterns across users.")
    st.markdown(
        "- **Data:** MovieLens \"latest small\": 100,836 ratings of 9,742 movies by 610 users, on a "
        "0.5-5 star scale.\n"
        "- **Split:** 70% of ratings train the models, 15% are for validation (tuning and "
        "comparison), and 15% are a test set.\n"
        "- **How it's scored:** each user's top-rated training movie is the \"seed\". The model "
        "recommends 10 movies for it, and we check how many appear among the movies that user rated "
        "4+ stars in the validation set.\n"
        "- **Three models:** a popularity baseline, item-based KNN on rating patterns, and a hybrid "
        "KNN that also uses genres."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ratings", f"{len(ratings):,}")
    col2.metric("Users", f"{ratings[USER_COL].nunique():,}")
    col3.metric("Movies", f"{len(movies):,}")
    col4.metric("Movies in the KNN", f"{models['matrix'].shape[0]:,}", help="Movies with 5+ training ratings")
    section("Train, validation, and test sets")
    table(pd.DataFrame({"Set": ["Training", "Validation", "Testing"],
                        "Ratings": [len(train), len(val), len(test)]}),
          column_config={"Ratings": st.column_config.NumberColumn(format="localized")})


def show_baseline_knn(movies, models, results):
    section("1. Popularity baseline",
            "Recommends the highest-rated movies with at least 5 ratings, the same list for everyone. "
            "Any real model has to beat this.")
    table(recommend_popular_movies(models["popular"], movies).round(2), column_config={
        "title": "Title", "avg_rating": "Average rating", "rating_count": "Ratings"})

    section("2. Item-based KNN",
            "Each movie is a vector of the ratings every user gave it. KNN with cosine similarity "
            "finds the movies whose rating patterns are closest. Example: the 10 movies most similar "
            "to Toy Story (1995).")
    matrix = models["matrix"]
    distances, indices = models["knn"].kneighbors(models["sparse"][matrix.index.get_loc(1)], n_neighbors=11)
    neighbors = pd.DataFrame({ITEM_COL: matrix.index[indices[0]], "similarity": 1 - distances[0]})
    neighbors = neighbors[neighbors[ITEM_COL] != 1].head(10).merge(movies[[ITEM_COL, "title"]], on=ITEM_COL)
    table(neighbors[["title", "similarity"]].round(3), column_config={"title": "Title"})

    section("3. Baseline vs. KNN on validation users",
            "Recall@10: share of each user's liked validation movies that were recommended. "
            "Hit Rate@10: share of users who got at least one movie they liked.")
    comparison = pd.DataFrame([results["baseline"], results["knn"]],
                              index=["Popularity baseline", "Item-based KNN"]).reset_index(names="Model")
    table(comparison.round(3))
    fig = px.bar(comparison, x="Model", y="Hit Rate@10", text_auto=".1%", title="Hit Rate@10")
    fig.update_traces(marker_color=BLUE)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig, "", "Hit Rate@10"), width="stretch")


def show_tuning(results):
    section("Tuning the KNN",
            "Three settings were tested: the minimum ratings a movie needs to be included, the "
            "minimum users who rated both movies, and how many neighbors to consider. 27 "
            "combinations in all, sorted by hit rate.")
    tuning = results["tuning"]
    heatmap = tuning[tuning["candidates"] == max(CANDIDATE_OPTIONS)].pivot(
        index="min_movie_ratings", columns="min_shared_users", values="hit_rate")
    fig = px.imshow(heatmap, text_auto=".1%", color_continuous_scale="Blues", aspect="auto",
                    title="KNN Hit Rate@10 by rating thresholds (200 candidate neighbors)")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    st.plotly_chart(style(fig, "Minimum shared users", "Minimum movie ratings"), width="stretch")
    table(tuning.round(3), height=360)

    best = tuning.iloc[0]
    section("Best configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Minimum movie ratings", int(best["min_movie_ratings"]))
    col2.metric("Minimum shared users", int(best["min_shared_users"]))
    col3.metric("Candidate neighbors", int(best["candidates"]))
    table(pd.DataFrame({"Model": ["Original KNN", "Tuned KNN"],
                        "Hit Rate@10": [results["knn"]["Hit Rate@10"], best["hit_rate"]],
                        "Recall@10": [results["knn"]["Recall@10"], best["recall"]]}).round(3))


def show_hybrid(results):
    section("Hybrid KNN: ratings + genres",
            "Similarity is a weighted mix of rating-pattern similarity and genre similarity. More "
            "weight on genres finds movies of the same kind; more weight on ratings finds movies the "
            "same people liked. Four weights were tested.")
    hybrid = results["hybrid"]
    fig = px.line(hybrid, x="rating_weight", y="hit_rate", markers=True,
                  title="Hit Rate@10 by weight on rating patterns")
    fig.update_traces(line_color=GOLD, line_width=2.5, marker_size=9)
    fig.update_xaxes(tickvals=RATING_WEIGHTS, tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig, "Weight on rating patterns (the rest is genres)", "Hit Rate@10"),
                    width="stretch")
    table(hybrid[["rating_weight", "genre_weight", "recall", "hit_rate", "avg_recommendations"]].round(3))


def show_comparison(movies, models, results):
    best_hybrid = results["hybrid"].loc[results["hybrid"]["hit_rate"].idxmax()]
    comparison = pd.DataFrame({
        "Model": ["Popularity baseline", "Ratings-only KNN", "Hybrid KNN"],
        "Recall@10": [results["baseline"]["Recall@10"], results["rating"]["recall"], best_hybrid["recall"]],
        "Hit Rate@10": [results["baseline"]["Hit Rate@10"], results["rating"]["hit_rate"],
                        best_hybrid["hit_rate"]],
    })
    section("Model performance",
            f"The ratings-only KNN uses the tuned settings. The hybrid uses its best weight "
            f"({best_hybrid['rating_weight']:.0%} ratings, {best_hybrid['genre_weight']:.0%} genres).")
    table(comparison.round(3))
    fig = px.bar(comparison, x="Model", y="Hit Rate@10", text_auto=".1%", title="Hit Rate@10 by model")
    fig.update_traces(marker_color=BLUE)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig, "", "Hit Rate@10"), width="stretch")

    section("Recommendations for Toy Story (1995)", "The same seed movie, three models.")
    table(pd.DataFrame({
        "Popularity baseline": recommend_popular_movies(models["popular"], movies, exclude_movie_id=1)["title"],
        "Ratings-only KNN": recommend_tuned_knn(models, 1, movies)["title"],
        "Hybrid KNN": recommend_hybrid(models, 1, movies)["title"],
    }))


def show_try_it(movies, models):
    section("Get recommendations",
            "Pick a movie (one with at least 10 training ratings) and compare what each model "
            "recommends.")
    eligible = movies[movies[ITEM_COL].isin(models["tuned_matrix"].index)].sort_values("title")
    titles = eligible["title"].tolist()
    chosen = st.selectbox("Movie", titles, index=titles.index("Toy Story (1995)"))
    movie_id = int(eligible.loc[eligible["title"] == chosen, ITEM_COL].iloc[0])
    weight = st.slider("Hybrid: weight on rating patterns", 0.0, 1.0, BEST_HYBRID_WEIGHT, step=0.1,
                       help="The rest of the weight goes to genre similarity. 0.9 scored best.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Ratings-only KNN**")
        table(recommend_tuned_knn(models, movie_id, movies).round(3),
              column_config={"title": "Title", "shared_users": "Shared users"})
    with right:
        st.markdown(f"**Hybrid KNN** ({weight:.0%} ratings, {1 - weight:.0%} genres)")
        table(recommend_hybrid(models, movie_id, movies, weight).round(3),
              column_config={"title": "Title", "shared_users": "Shared users"})


def render():
    page_header("Movie recommender",
                "Item-based KNN on MovieLens ratings: movies people rated alike, plus a version that "
                "also uses genres.")
    if missing_files(FILES, "Run `notebooks/movie_rate_data.ipynb` to download and save them."):
        return
    ratings, movies, train, val, test = load_data()
    models = fitted_models()
    choice = subpages(["Overview", "Baseline and KNN", "Tuning", "Hybrid model", "Comparison", "Try it"],
                      key="recommender_section")
    if choice == "Overview":
        show_overview(ratings, movies, train, val, test, models)
    elif choice == "Baseline and KNN":
        show_baseline_knn(movies, models, evaluation_results())
    elif choice == "Tuning":
        show_tuning(evaluation_results())
    elif choice == "Hybrid model":
        show_hybrid(evaluation_results())
    elif choice == "Comparison":
        show_comparison(movies, models, evaluation_results())
    else:
        show_try_it(movies, models)
