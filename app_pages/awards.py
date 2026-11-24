"""Oscar prediction page: logistic regression for a Best Picture nomination.

Same steps and settings as notebooks/award_modeling.ipynb.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app_pages.ui import (BLUE, GOLD, PROCESSED_DIRECTORY, RAW_DIRECTORY, missing_files,
                          page_header, section, style, subpages, table)

FIRST_YEAR = 1950
C_VALUES = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]
REVIEW_COUNTS = ["positivity_n", "negativity_n", "performance_mentions_n", "performance_tone_n",
                 "production_mentions_n", "production_tone_n", "effects_mentions_n", "effects_tone_n",
                 "humor_mentions_n", "humor_tone_n", "aesthetics_mentions_n", "aesthetics_tone_n"]
REVIEW_SCORES = ["positivity", "negativity", "performance_mentions", "performance_tone",
                 "production_mentions", "production_tone", "effects_mentions", "effects_tone",
                 "humor_mentions", "humor_tone", "aesthetics_mentions", "aesthetics_tone"]
NUMERIC = ["averageRating", "numVotes", "runtimeMinutes", "popularity", "cast_count",
           "crew_count", "n_directors", "vfx_crew", "review_total", "budget", "revenue"]
REVIEW = ["mean_rating", "has_reviews"] + REVIEW_COUNTS + REVIEW_SCORES
PRE_NUMERIC = ["budget", "cast_count", "crew_count", "n_directors", "vfx_crew", "runtimeMinutes"]
TMDB_FIELDS = ["imdb_id", "popularity", "cast_count", "crew_count", "n_directors",
               "vfx_crew", "review_total", "budget", "revenue", "vote_average", "vote_count"]

FILES = [PROCESSED_DIRECTORY / "oscars_cleaned.csv", PROCESSED_DIRECTORY / "imdb_cleaned.csv",
         RAW_DIRECTORY / "tmdb_movies.csv", PROCESSED_DIRECTORY / "review_features_by_movie.csv"]


@st.cache_data(show_spinner="Building the candidate pool and imputing missing values...")
def build_data():
    oscars = pd.read_csv(PROCESSED_DIRECTORY / "oscars_cleaned.csv")
    imdb = pd.read_csv(PROCESSED_DIRECTORY / "imdb_cleaned.csv", dtype={"tconst": "str"})

    movies = pd.read_csv(RAW_DIRECTORY / "tmdb_movies.csv", usecols=["tmdb_id", "imdb_id"])
    review_features = pd.read_csv(PROCESSED_DIRECTORY / "review_features_by_movie.csv")
    bridge = movies.dropna(subset=["imdb_id"]).drop_duplicates("imdb_id")[["imdb_id", "tmdb_id"]]
    review_by_imdb = bridge.merge(review_features, on="tmdb_id", how="inner")

    oscars = oscars[oscars["Year"] >= FIRST_YEAR].copy()
    bp = oscars[(oscars["CanonicalCategory"] == "BEST PICTURE") & oscars["FilmId"].notna()]
    bp = bp[~bp["FilmId"].str.contains(r"\|")]
    nominated_ids = set(bp["FilmId"].str.strip())

    films = imdb[imdb["startYear"].between(1950, 2025)].copy()
    films["startYear"] = films["startYear"].astype(int)
    films["nominated"] = films["tconst"].isin(nominated_ids).astype(int)
    model_rows = films.dropna(subset=["runtimeMinutes", "averageRating", "numVotes", "genres"]).copy()

    tmdb = pd.read_csv(RAW_DIRECTORY / "tmdb_movies.csv", usecols=TMDB_FIELDS)
    tmdb[["budget", "revenue"]] = tmdb[["budget", "revenue"]].replace(0, np.nan)
    tmdb = tmdb.dropna(subset=["imdb_id"])
    pool = model_rows.merge(tmdb, left_on="tconst", right_on="imdb_id", how="inner", validate="m:1")
    pool = pool.merge(review_by_imdb.drop(columns=["title"]), on="imdb_id", how="left")

    funnel = pd.DataFrame([
        {"Step": "IMDb movies 1950-2025 with complete basics", "Films": len(model_rows),
         "Best Picture nominees": int(model_rows["nominated"].sum())},
        {"Step": "...that have a TMDB row (100+ TMDB votes)", "Films": len(pool),
         "Best Picture nominees": int(pool["nominated"].sum())},
        {"Step": "...that also have review features", "Films": int(pool["reviews"].notna().sum()),
         "Best Picture nominees": int(pool.loc[pool["reviews"].notna(), "nominated"].sum())},
    ])
    funnel["Nominee share"] = funnel["Best Picture nominees"] / funnel["Films"]

    # Structural missingness: no reviews means zero mentions, recorded with a flag
    pool_raw = pool.copy()
    pool["has_reviews"] = (pool["review_total"] > 0).astype(int)
    pool[REVIEW_COUNTS] = pool[REVIEW_COUNTS].fillna(0)
    pool = pool.drop(columns=["reviews"])

    # Regression imputation with noise for the rest
    to_impute = ["budget", "revenue", "mean_rating"] + REVIEW_SCORES
    candidates = ["averageRating", "numVotes", "runtimeMinutes", "popularity", "cast_count",
                  "crew_count", "n_directors", "vfx_crew", "review_total", "has_reviews"] + \
                 REVIEW_COUNTS + to_impute
    rng = np.random.default_rng(42)
    log = []
    for column in to_impute:
        missing = pool[column].isna()
        if missing.sum() == 0:
            continue
        predictors = [c for c in candidates if c != column and pool[c].notna().all()]
        observed = pool.loc[~missing]
        mean_before = pool[column].mean()
        fit = sm.OLS(observed[column], sm.add_constant(observed[predictors], has_constant="add")).fit()
        predicted = fit.predict(sm.add_constant(pool.loc[missing, predictors], has_constant="add"))
        noise = rng.normal(0, fit.resid.std(ddof=1), size=int(missing.sum()))
        values = (predicted + noise).clip(pool[column].min(), pool[column].max())
        pool[column] = pool[column].where(~missing, values)
        log.append({"Column": column, "Imputed rows": int(missing.sum()), "R-squared": fit.rsquared,
                    "Mean before": mean_before, "Mean after": pool[column].mean()})

    genre_flags = pool["genres"].str.get_dummies(sep=",")
    top_genres = genre_flags.sum().sort_values(ascending=False).head(12).index.tolist()
    pool[top_genres] = genre_flags[top_genres].astype(int)
    data = pool.dropna(subset=NUMERIC + REVIEW).copy()

    # Pre-production data: budget re-imputed from pre-production predictors only
    exante = pool_raw[["tconst", "startYear", "nominated"] + PRE_NUMERIC + ["genres"]].copy()
    pre_genre_flags = exante["genres"].str.get_dummies(sep=",")
    pre_genres = pre_genre_flags.sum().sort_values(ascending=False).head(12).index.tolist()
    exante[pre_genres] = pre_genre_flags[pre_genres].astype(int)
    rng = np.random.default_rng(42)
    pre_log = []
    for column in PRE_NUMERIC:
        missing = exante[column].isna()
        if missing.sum() == 0:
            continue
        predictors = [c for c in PRE_NUMERIC + pre_genres if c != column and exante[c].notna().all()]
        observed = exante.loc[~missing]
        fit = sm.OLS(observed[column], sm.add_constant(observed[predictors], has_constant="add")).fit()
        predicted = fit.predict(sm.add_constant(exante.loc[missing, predictors], has_constant="add"))
        noise = rng.normal(0, fit.resid.std(ddof=1), size=int(missing.sum()))
        exante[column] = exante[column].where(
            ~missing, (predicted + noise).clip(exante[column].min(), exante[column].max()))
        pre_log.append({"Column": column, "Imputed rows": int(missing.sum()), "R-squared": fit.rsquared})

    return {"funnel": funnel, "imputation": pd.DataFrame(log), "pre_imputation": pd.DataFrame(pre_log),
            "data": data, "top_genres": top_genres, "exante": exante, "pre_genres": pre_genres,
            "no_reviews": int((pool["has_reviews"] == 0).sum())}


def make_model(numeric, binary, C):
    return Pipeline(steps=[
        ("preprocess", ColumnTransformer(transformers=[
            ("num", StandardScaler(), numeric),
            ("bin", "passthrough", binary),
        ])),
        ("model", LogisticRegression(C=C, l1_ratio=0, solver="lbfgs", max_iter=5000)),
    ])


def evaluate(X, y, numeric, binary):
    """Pick C by 10-fold CV log loss, then fit on the training split and score the holdout once."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    rows = []
    for C in C_VALUES:
        result = cross_validate(make_model(numeric, binary, C), X, y, cv=cv,
                                scoring=["neg_log_loss", "roc_auc"], n_jobs=-1)
        rows.append({"C": C, "log_loss": -result["test_neg_log_loss"].mean(),
                     "sd": result["test_neg_log_loss"].std(), "AUC": result["test_roc_auc"].mean()})
    cv_results = pd.DataFrame(rows)
    best_C = cv_results.loc[cv_results["log_loss"].idxmin(), "C"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = make_model(numeric, binary, best_C).fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    y_hat = (p_test >= 0.5).astype(int)
    coefficients = pd.Series(model.named_steps["model"].coef_[0], index=numeric + binary)
    odds = pd.DataFrame({"Predictor": coefficients.index, "Coefficient": coefficients.values,
                         "Odds ratio": np.exp(coefficients.values)})
    odds = odds.reindex(odds["Coefficient"].abs().sort_values(ascending=False).index)
    holdout = {
        "rows": len(X_test), "nominees": int(y_test.sum()),
        "base_rate_log_loss": log_loss(y_test, np.full(len(y_test), y_train.mean())),
        "cv_base_rate_log_loss": log_loss(y, np.full(len(y), y.mean())),
        "log_loss": log_loss(y_test, p_test), "AUC": roc_auc_score(y_test, p_test),
        "accuracy": accuracy_score(y_test, y_hat), "precision": precision_score(y_test, y_hat),
        "recall": recall_score(y_test, y_hat), "confusion": confusion_matrix(y_test, y_hat),
    }
    return {"cv": cv_results, "best_C": best_C, "holdout": holdout, "odds": odds, "model": model}


@st.cache_resource(show_spinner="Fitting both logistic models with 10-fold CV (about a minute)...")
def model_results():
    built = build_data()
    data, exante = built["data"], built["exante"]
    full = evaluate(data[NUMERIC + REVIEW + built["top_genres"]], data["nominated"],
                    NUMERIC + REVIEW, built["top_genres"])
    pre = evaluate(exante[PRE_NUMERIC + built["pre_genres"]], exante["nominated"],
                   PRE_NUMERIC, built["pre_genres"])
    return {"full": full, "pre": pre}


def cv_chart(cv_results, best_C, title):
    fig = go.Figure(go.Scatter(x=cv_results["C"], y=cv_results["log_loss"], mode="lines+markers",
                               line={"color": GOLD, "width": 2.5}, marker={"size": 8},
                               error_y={"type": "data", "array": cv_results["sd"], "color": "#d9c27a"}))
    fig.add_vline(x=best_C, line_dash="dot", line_color=BLUE,
                  annotation_text=f"best C = {best_C:g}", annotation_position="top")
    fig.update_xaxes(type="log", tickvals=C_VALUES,
                     ticktext=[f"1/{round(1 / c)}" if c < 1 else f"{c:g}" for c in C_VALUES])
    fig.update_layout(title=title)
    return style(fig, "C (smaller = stronger ridge penalty, log scale)", "Mean CV log loss")


def show_holdout(result):
    holdout = result["holdout"]
    st.markdown(f"Fit on the training 80% with C = {result['best_C']:g}, then scored once on the "
                f"{holdout['rows']:,} held-out films ({holdout['nominees']} nominees).")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Log loss", f"{holdout['log_loss']:.4f}",
                help=f"Base rate (always predict the share of nominees): {holdout['base_rate_log_loss']:.4f}")
    col2.metric("AUC", f"{holdout['AUC']:.3f}")
    col3.metric("Accuracy", f"{holdout['accuracy']:.1%}")
    col4.metric("Precision", f"{holdout['precision']:.1%}")
    col5.metric("Recall", f"{holdout['recall']:.1%}")
    confusion = holdout["confusion"]
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Confusion matrix (threshold 0.5)**")
        table(pd.DataFrame({"Actual": ["Not nominated", "Nominated"],
                            "Predicted: not nominated": confusion[:, 0],
                            "Predicted: nominated": confusion[:, 1]}))
        st.caption("Nominees are only about 2% of films, so accuracy is high even for a weak model. "
                   "Log loss and AUC are the better measures here.")
    with right:
        odds = result["odds"].head(15)
        fig = px.bar(odds.iloc[::-1], x="Coefficient", y="Predictor", orientation="h",
                     title="15 strongest predictors (standardized coefficients)")
        fig.update_traces(marker_color=[GOLD if c > 0 else BLUE for c in odds["Coefficient"].iloc[::-1]])
        st.plotly_chart(style(fig, "Coefficient (positive = more likely nominated)", ""), width="stretch")
    with st.expander("All coefficients and odds ratios"):
        table(result["odds"].round(3))


def show_overview(built, results):
    section("The question",
            "Can we predict which films get nominated for Best Picture, from data about the film "
            "and how people reviewed it?")
    st.markdown(
        "- **Target:** nominated for Best Picture (win or lose), ceremonies 1950-2025.\n"
        "- **Candidate films:** IMDb movies with a TMDB record (100+ TMDB votes), so obscure films "
        "are left out.\n"
        "- **Model:** logistic regression with a ridge penalty, chosen by 10-fold cross-validated "
        "log loss, then tested once on a 20% holdout.\n"
        "- **Two versions:** a full model that uses everything, including ratings and reviews, and "
        "a pre-production model that only uses what a producer would know before filming."
    )
    data = built["data"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidate films", f"{len(data):,}")
    col2.metric("Best Picture nominees", f"{int(data['nominated'].sum()):,}")
    col3.metric("Full model AUC (holdout)", f"{results['full']['holdout']['AUC']:.3f}")
    col4.metric("Pre-production AUC (holdout)", f"{results['pre']['holdout']['AUC']:.3f}")
    section("Building the candidate pool")
    table(built["funnel"], column_config={
        "Films": st.column_config.NumberColumn(format="localized"),
        "Nominee share": st.column_config.NumberColumn(format="percent")})


def show_data(built):
    section("1. Reviews: structural missing values",
            f"A film with no reviews has zero review mentions by definition. That's "
            f"{built['no_reviews']:,} films: their mention counts are set to 0 and a has_reviews "
            f"flag marks them.")
    section("2. Regression imputation with noise",
            "For budget, revenue, mean review rating, and the twelve review sentiment scores: fit an "
            "OLS on the rows where the value is known, predict the missing rows, and add random "
            "noise with the same spread as the model's errors, so imputed values aren't "
            "unrealistically tidy.")
    # Budget and revenue means are in dollars, the rest are scores, so format each value by its size
    readable = lambda value: f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"
    table(built["imputation"].style.format(readable, subset=["Mean before", "Mean after"]),
          column_config={
              "Imputed rows": st.column_config.NumberColumn(format="localized"),
              "R-squared": st.column_config.NumberColumn(format="%.3f")})
    section("3. Pre-production imputation",
            "The pre-production model re-imputes budget using only pre-production inputs, so no "
            "information from after release leaks in.")
    table(built["pre_imputation"].round(3))


def show_models(results, built):
    st.markdown(
        "Both models use **the same steps**: pick the ridge penalty C with 10-fold CV, then fit on "
        "the training 80% and score once on the same 20% holdout. **Only the inputs differ**, so "
        "the gap between them shows how much a film's reception adds."
    )
    table(pd.DataFrame({
        "": ["Question", "Inputs", "Holdout AUC"],
        "Full model": ["Can we spot a nominee once it's out?",
                       f"{len(NUMERIC + REVIEW + built['top_genres'])}: everything, including ratings and reviews",
                       f"{results['full']['holdout']['AUC']:.3f}"],
        "Pre-production model": ["Can we spot one before it's made?",
                                 f"{len(PRE_NUMERIC + built['pre_genres'])}: budget, credits, runtime, and genres only",
                                 f"{results['pre']['holdout']['AUC']:.3f}"],
    }))
    choice = st.radio("Model", ["Full model", "Pre-production model"], horizontal=True,
                      key="awards_model")
    key = "full" if choice == "Full model" else "pre"
    result = results[key]

    if key == "full":
        section("1. Predictors",
                f"{len(NUMERIC + REVIEW + built['top_genres'])} predictors: IMDb rating, votes and "
                "runtime; TMDB popularity, credits, review count, budget and revenue; review volume and "
                "sentiment; and the 12 most common genres.")
    else:
        section("1. Predictors",
                "Only what a producer could know before shooting: budget, cast size, crew size, number "
                "of directors, visual effects crew, runtime, and the 12 most common genres.")
        st.caption("Caveat: the candidate pool is still films with 100+ TMDB votes, which is itself a "
                   "fact about later reception. The predictors are pre-production; the pool is not.")
    section("2. Choosing the penalty with 10-fold CV",
            f"Mean log loss across 10 stratified folds for each C. The base-rate log loss to beat is "
            f"{result['holdout']['cv_base_rate_log_loss']:.4f}.")
    st.plotly_chart(cv_chart(result["cv"], result["best_C"], f"{choice}: CV log loss by penalty"),
                    width="stretch")
    with st.expander("CV results for every C"):
        table(result["cv"].round(6))
    section("3. Holdout performance and coefficients")
    show_holdout(result)


def show_comparison(results):
    section("Full vs. pre-production model", "Both scored on the same 20% holdout.")
    rows = []
    for name, key in [("Full model", "full"), ("Pre-production model", "pre")]:
        holdout = results[key]["holdout"]
        rows.append({"Model": name, "Best C": results[key]["best_C"],
                     "Mean CV log loss": results[key]["cv"]["log_loss"].min(),
                     "Holdout log loss": holdout["log_loss"],
                     "Base-rate log loss": holdout["base_rate_log_loss"], "Holdout AUC": holdout["AUC"],
                     "Precision": holdout["precision"], "Recall": holdout["recall"]})
    comparison = pd.DataFrame(rows)
    table(comparison.round(4))
    fig = px.bar(comparison.melt(id_vars="Model", value_vars=["Holdout log loss", "Base-rate log loss"]),
                 x="Model", y="value", color="variable", barmode="group", text_auto=".3f",
                 color_discrete_sequence=[GOLD, "#b5b4ae"], title="Holdout log loss vs. the base rate")
    st.plotly_chart(style(fig, "", "Log loss (lower is better)"), width="stretch")
    st.markdown("Both models beat the base rate. The full model does much better because ratings and "
                "reviews carry a lot of information, but it can only be used after a film is out.")


def show_try_it(built, results):
    section("Pre-production nomination chance",
            "Describe a film before it's made, and the pre-production model estimates the chance it "
            "gets a Best Picture nomination.")
    exante = built["exante"]
    model = results["pre"]["model"]
    left, right = st.columns([1.2, 1])
    with left:
        budget = st.number_input("Budget (millions of dollars)", 0.5, 400.0,
                                 float(round(exante["budget"].median() / 1e6, 1)), step=1.0)
        runtime = st.slider("Runtime (minutes)", 70, 240, int(exante["runtimeMinutes"].median()))
        cast = st.slider("Credited cast", 1, 150, int(exante["cast_count"].median()))
        crew = st.slider("Credited crew", 1, 600, int(exante["crew_count"].median()))
        directors = st.slider("Directors", 1, 4, 1)
        vfx = st.slider("Visual effects crew", 0, 300, int(exante["vfx_crew"].median()))
        genres = st.multiselect("Genres (up to 3)", built["pre_genres"], default=["Drama"],
                                max_selections=3)
    film = pd.DataFrame([{"budget": budget * 1e6, "cast_count": cast, "crew_count": crew,
                          "n_directors": directors, "vfx_crew": vfx, "runtimeMinutes": runtime,
                          **{genre: int(genre in genres) for genre in built["pre_genres"]}}])
    probability = model.predict_proba(film[PRE_NUMERIC + built["pre_genres"]])[0, 1]
    base_rate = exante["nominated"].mean()
    with right:
        st.metric("Chance of a Best Picture nomination", f"{probability:.1%}",
                  delta=f"{probability / base_rate:.1f}x the average film", delta_color="off")
        st.caption(f"The average candidate film has a {base_rate:.1%} chance.")


def render():
    page_header("Oscar prediction",
                "Logistic regression for a Best Picture nomination, with everything we know about a "
                "film and with pre-production information only.")
    if missing_files(FILES, "Run `notebooks/tmdb_pull.ipynb` and then `notebooks/review_features.ipynb`, "
                            "or copy the files from a teammate."):
        return
    built = build_data()
    results = model_results()
    choice = subpages(["Overview", "Data and imputation", "Models", "Comparison", "Try it"],
                      key="awards_section")
    if choice == "Overview":
        show_overview(built, results)
    elif choice == "Data and imputation":
        show_data(built)
    elif choice == "Models":
        show_models(results, built)
    elif choice == "Comparison":
        show_comparison(results)
    else:
        show_try_it(built, results)
