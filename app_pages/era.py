"""Era comparison page: are older movies better once eras are compared on fair terms?

Same steps as notebooks/era_comparison.ipynb.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from patsy import dmatrices
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from app_pages.ui import (PROCESSED_DIRECTORY, missing_files, page_header, section, style,
                          subpages, table)

MIN_VOTES = 1_000
TOP_N = 1_000  # movies per era in the overview table
ERAS = ["Old", "Middle", "New"]
ERA_LABELS = {"Old": "Old (before 1980)", "Middle": "Middle (1980-1999)", "New": "New (2000+)"}
# Eras are ordered, so one hue from dark (oldest) to light (newest)
ERA_COLORS = {"Old": "#0b3d78", "Middle": "#2a78d6", "New": "#86b6ec"}
FILES = [PROCESSED_DIRECTORY / "imdb_cleaned.csv", PROCESSED_DIRECTORY / "oscars_cleaned.csv"]
FEATURE_LABELS = {"popularity_pct": "Popularity within era (0 to 1)", "runtime_hours": "Runtime (per hour)",
                  "oscar_nominated": "Oscar nominated", "SciFi": "Sci-Fi"}


@st.cache_data(show_spinner="Loading IMDb movies...")
def load_movies():
    imdb = pd.read_csv(PROCESSED_DIRECTORY / "imdb_cleaned.csv")
    oscars = pd.read_csv(PROCESSED_DIRECTORY / "oscars_cleaned.csv")

    film_nominations = oscars[~oscars["Class"].isin(["SciTech", "Special"])].dropna(subset=["FilmId"])
    oscar_nominations = (
        film_nominations.assign(FilmId=film_nominations["FilmId"].str.split("|"))
        .explode("FilmId")
        .groupby("FilmId")
        .size()
    )

    movies = imdb[
        (imdb["numVotes"] >= MIN_VOTES) & imdb["startYear"].between(1920, 2025)
    ].dropna(subset=["genres", "runtimeMinutes"]).copy()
    movies["era"] = pd.cut(movies["startYear"], bins=[0, 1979, 1999, 3000], labels=ERAS)
    movies["popularity_pct"] = movies.groupby("era", observed=True)["numVotes"].rank(pct=True)
    movies["runtime_hours"] = movies["runtimeMinutes"].clip(upper=240) / 60
    movies["oscar_nominated"] = movies["tconst"].isin(oscar_nominations.index).astype(int)

    top_genres = movies["genres"].str.split(",").explode().value_counts().head(12).index
    genre_columns = []
    for genre in top_genres:
        column = genre.replace("-", "")
        movies[column] = movies["genres"].str.split(",").apply(lambda genres: genre in genres).astype(int)
        genre_columns.append(column)
    movies["popularity_decile"] = np.ceil(movies["popularity_pct"] * 10).astype(int)
    return movies, genre_columns


def formulas(features):
    terms = " + ".join(features)
    return {
        "No era": f"averageRating ~ {terms}",
        "Era": f"averageRating ~ C(era, Treatment('New')) + {terms}",
        "Era x features": f"averageRating ~ C(era, Treatment('New')) * ({terms})",
    }


def design(formula, data):
    """Outcome and design matrix from a formula, without the intercept column (sklearn adds one)."""
    y, X = dmatrices(formula, data, return_type="dataframe")
    return y["averageRating"], X.drop(columns="Intercept")


def split(movies):
    return train_test_split(movies, test_size=0.2, random_state=42, stratify=movies["era"])


def readable_tukey(tukey_table):
    """Tukey rows as 'older era vs. newer era', with the gap as older minus newer."""
    rows = []
    for _, row in tukey_table.iterrows():
        group1, group2 = row["group1"], row["group2"]
        # statsmodels reports meandiff = mean(group2) - mean(group1); flip so it's older minus newer
        if ERAS.index(group1) < ERAS.index(group2):
            older, newer, sign = group1, group2, -1
        else:
            older, newer, sign = group2, group1, 1
        low, high = sorted([sign * row["lower"], sign * row["upper"]])
        rows.append({"Comparison": f"{older} vs. {newer}",
                     "Difference in average rating": sign * row["meandiff"],
                     "95% CI": f"{low:+.2f} to {high:+.2f}",
                     "Real difference?": "Yes" if row["reject"] else "No"})
    order = {"Old vs. New": 0, "Old vs. Middle": 1, "Middle vs. New": 2}
    return pd.DataFrame(rows).sort_values("Comparison", key=lambda c: c.map(order))


@st.cache_data(show_spinner="Comparing the eras...")
def era_comparison():
    movies, _ = load_movies()
    # Same number of movies from each era: its 1,000 most-voted
    top = movies.sort_values("numVotes", ascending=False).groupby("era", observed=True).head(TOP_N)
    summary = top.groupby("era", observed=True).agg(
        median_rating=("averageRating", "median"),
        median_votes=("numVotes", "median"),
        oscar_nominated=("oscar_nominated", "mean"),
    ).reset_index()
    era_sizes = movies.groupby("era", observed=True).size()
    summary.insert(1, "movies_in_era", summary["era"].map(era_sizes).astype(int))
    summary.insert(2, "share_of_era", TOP_N / summary["movies_in_era"])
    percentiles = (
        movies.groupby("era", observed=True)["averageRating"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
        .unstack()
        .rename(columns=lambda p: f"{p:.0%}")
    )
    by_popularity = (
        movies.groupby(["popularity_decile", "era"], observed=True)["averageRating"].median().unstack()
    )

    tests = {}
    for label, data in [("All movies", movies),
                        ("Each era's 10% most-voted", movies[movies["popularity_decile"] == 10])]:
        anova = sm.stats.anova_lm(smf.ols("averageRating ~ C(era)", data=data).fit())
        tukey = pairwise_tukeyhsd(data["averageRating"], data["era"].astype(str))
        tukey_table = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
        tests[label] = {"movies": len(data), "F": anova["F"].iloc[0], "p": anova["PR(>F)"].iloc[0],
                        "tukey": readable_tukey(tukey_table)}
    return summary, percentiles, by_popularity, tests


@st.cache_data(show_spinner="Fitting the regression models (about a minute the first time)...")
def model_results():
    movies, genre_columns = load_movies()
    features = ["popularity_pct", "runtime_hours", "oscar_nominated", *genre_columns]
    model_formulas = formulas(features)
    train, test = split(movies)
    cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)

    feature_matrix = sm.add_constant(movies[features])
    vif = pd.DataFrame({
        "Feature": features,
        "VIF": [variance_inflation_factor(feature_matrix.values, i)
                for i in range(1, feature_matrix.shape[1])],
    }).sort_values("VIF", ascending=False)

    def cv_scores(estimator, X, y):
        scores = cross_validate(estimator, X, y, cv=cv_folds, scoring="neg_root_mean_squared_error")
        return -scores["test_score"]

    estimators = {
        "OLS": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "Lasso": LassoCV(cv=5, random_state=42),
    }
    y_train = train["averageRating"]
    rows = [{"Model": "Baseline", "Estimator": "Mean", "Inputs": 0,
             "CV RMSE": cv_scores(DummyRegressor(), train[features], y_train).mean()}]
    fold_scores = {}
    for model_name, formula in model_formulas.items():
        y, X = design(formula, train)
        for estimator_name, estimator in estimators.items():
            scores = cv_scores(make_pipeline(StandardScaler(), estimator), X, y)
            rows.append({"Model": model_name, "Estimator": estimator_name, "Inputs": X.shape[1],
                         "CV RMSE": scores.mean()})
        fold_scores[model_name] = cv_scores(LinearRegression(), X, y)
    cv_results = pd.DataFrame(rows)
    fold_scores = pd.DataFrame(fold_scores, index=[f"Fold {i}" for i in range(1, 6)])

    era_fit = smf.ols(model_formulas["Era"], data=train).fit()
    robust_fit = smf.ols(model_formulas["Era"], data=train).fit(cov_type="HC3")
    era_terms = [term for term in era_fit.params.index if term.startswith("C(era")]
    era_gap = pd.DataFrame({
        "Era": ["Old (before 1980)", "Middle (1980-1999)"],
        "Rating gap vs. new movies": era_fit.params[era_terms].values,
        "95% CI low": era_fit.conf_int().loc[era_terms, 0].values,
        "95% CI high": era_fit.conf_int().loc[era_terms, 1].values,
        "Robust 95% CI": [f"{low:.3f} to {high:.3f}"
                          for low, high in robust_fit.conf_int().loc[era_terms].values],
    })

    coefficients = []
    for era in ERAS:
        fit = smf.ols(f"averageRating ~ {' + '.join(features)}", data=train[train["era"] == era]).fit()
        interval = fit.conf_int()
        for feature in features:
            coefficients.append({"era": era, "feature": feature, "coefficient": fit.params[feature],
                                 "low": interval.loc[feature, 0], "high": interval.loc[feature, 1]})
    coefficients = pd.DataFrame(coefficients)

    y_train_full, X_train_full = design(model_formulas["Era x features"], train)
    y_test, X_test = design(model_formulas["Era x features"], test)
    final_model = LinearRegression().fit(X_train_full, y_train_full)
    predictions = final_model.predict(X_test)
    holdout = {
        "baseline_rmse": root_mean_squared_error(y_test, np.full(len(y_test), y_train.mean())),
        "rmse": root_mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "predicted": predictions,
        "residuals": (y_test - predictions).to_numpy(),
    }

    X_scaled = StandardScaler().fit_transform(X_train_full)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X_scaled, y_train_full)
    lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y_train_full)
    penalties = {"ridge_alpha": ridge.alpha_, "lasso_alpha": lasso.alpha_,
                 "lasso_zeroed": int((lasso.coef_ == 0).sum()), "inputs": X_train_full.shape[1],
                 "train_movies": len(train)}

    assumptions = {"bp_p": het_breuschpagan(era_fit.resid, era_fit.model.exog)[1],
                   "skew": era_fit.resid.skew()}

    cutoffs = []
    for old_end, middle_end in [(1969, 1999), (1979, 1999), (1989, 2009)]:
        recut = movies.copy()
        recut["era"] = pd.cut(recut["startYear"], bins=[0, old_end, middle_end, 3000], labels=ERAS)
        recut["popularity_pct"] = recut.groupby("era", observed=True)["numVotes"].rank(pct=True)
        fit = smf.ols(model_formulas["Era"], data=recut).fit()
        cutoffs.append({"Old": f"before {old_end + 1}", "Middle": f"{old_end + 1}-{middle_end}",
                        "New": f"{middle_end + 1}+", "Old gap vs. new": fit.params[era_terms[0]],
                        "Middle gap vs. new": fit.params[era_terms[1]]})

    return {"features": features, "vif": vif, "cv": cv_results, "folds": fold_scores,
            "era_gap": era_gap, "coefficients": coefficients, "holdout": holdout,
            "penalties": penalties, "assumptions": assumptions, "cutoffs": pd.DataFrame(cutoffs)}


@st.cache_resource(show_spinner="Fitting the prediction model...")
def prediction_model():
    """The era x features model, fit on the training movies, for the Try it section."""
    movies, genre_columns = load_movies()
    features = ["popularity_pct", "runtime_hours", "oscar_nominated", *genre_columns]
    train, _ = split(movies)
    return smf.ols(formulas(features)["Era x features"], data=train).fit(), genre_columns


def era_lines(data, x_title, y_title, title):
    fig = go.Figure()
    for era in ERAS:
        fig.add_trace(go.Scatter(x=data.index, y=data[era], mode="lines+markers", name=ERA_LABELS[era],
                                 line={"color": ERA_COLORS[era], "width": 2.5}, marker={"size": 8}))
    fig.update_layout(title=title, legend={"orientation": "h", "y": -0.25})
    return style(fig, x_title, y_title)


def show_overview(movies):
    section("The question",
            "Are older movies better than newer ones, once we compare them on fair terms? Here "
            "\"better\" means a higher IMDb rating.")
    st.markdown(
        "- **Data:** IMDb movies with at least 1,000 votes, released 1920-2025, joined to Oscar "
        "nominations by IMDb ID.\n"
        "- **Eras:** Old (before 1980), Middle (1980-1999), New (2000 and later).\n"
        "- **The catch:** people today mostly rate the old movies that are still remembered, so a "
        "plain average flatters old movies. Each movie's popularity is ranked *within its own era* "
        "to compare like with like.\n"
        "- **Two parts:** (1) compare the eras at the same level; (2) fit the same regression for "
        "every era to see what drives ratings, and test whether knowing the era improves predictions."
    )
    results = model_results()
    old_gap = results["era_gap"].iloc[0]
    best = results["cv"].loc[results["cv"]["CV RMSE"].idxmin()]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Movies analyzed", f"{len(movies):,}")
    col2.metric("Old vs. new, all else equal", f"+{old_gap['Rating gap vs. new movies']:.2f} points")
    col3.metric("Best CV RMSE", f"{best['CV RMSE']:.3f}", help=f"{best['Model']}, {best['Estimator']}")
    col4.metric("Holdout RMSE", f"{results['holdout']['rmse']:.3f}",
                help=f"Baseline: {results['holdout']['baseline_rmse']:.3f}")

    summary = era_comparison()[0]
    summary["era"] = summary["era"].map(ERA_LABELS)
    section(f"Each era's {TOP_N:,} most-voted movies",
            f"The eras have very different numbers of movies, so this compares the same number from "
            f"each: its {TOP_N:,} most-voted. That's a bigger slice of the old era than of the new "
            f"one (see \"Share of era\"), so old movies are held to a looser bar here.")
    table(summary, column_config={
        "era": "Era",
        "movies_in_era": st.column_config.NumberColumn("Movies in era", format="localized"),
        "share_of_era": st.column_config.NumberColumn("Share of era", format="percent"),
        "median_rating": st.column_config.NumberColumn("Median rating", format="%.1f"),
        "median_votes": st.column_config.NumberColumn("Median votes", format="localized"),
        "oscar_nominated": st.column_config.NumberColumn("Share Oscar-nominated", format="percent"),
    })


def show_comparison():
    _, percentiles, by_popularity, tests = era_comparison()

    section("1. At each percentile of rating",
            "Instead of one average per era, this lines up the worst-rated 10% of each era, the "
            "median, the best 10%, and so on. Old movies win clearly at the bottom; at the top the "
            "eras are nearly tied.")
    st.plotly_chart(era_lines(percentiles.T, "Percentile within the era", "IMDb rating",
                              "IMDb rating at each percentile, by era"), width="stretch")
    table(percentiles.reset_index().assign(era=lambda df: df["era"].map(ERA_LABELS)))

    section("2. At each level of popularity within the era",
            "Movies at the same popularity rank in their own era, from least voted (1) to most voted "
            "(10). Old movies rate higher at every level, and the gap is largest for the most "
            "popular movies.")
    st.plotly_chart(era_lines(by_popularity, "Popularity decile within the era (10 = most voted)",
                              "Median IMDb rating", "Median IMDb rating by popularity within the era"),
                    width="stretch")

    section("3. Are the differences real?",
            "ANOVA tests whether the eras' average ratings differ at all. Tukey's HSD then compares "
            "each pair of eras. A positive difference means the older era rates higher, in IMDb "
            "rating points. With this many movies every difference is significant, so the size of "
            "the gap is what matters.")
    for label, result in tests.items():
        p_text = "p < 0.001" if result["p"] < 0.001 else f"p = {result['p']:.3f}"
        gap = result["tukey"].iloc[0]["Difference in average rating"]
        st.markdown(f"**{label}** ({result['movies']:,} movies): ANOVA F = {result['F']:,.1f}, "
                    f"{p_text}. Old movies average **{gap:+.2f} points** compared with new ones.")
        table(result["tukey"], column_config={
            "Difference in average rating": st.column_config.NumberColumn(format="%+.2f")})


def show_models(results):
    section("1. Multicollinearity check")
    vif = results["vif"].assign(Feature=lambda df: df["Feature"].map(lambda f: FEATURE_LABELS.get(f, f)))
    left, right = st.columns([1, 2])
    with left:
        table(vif.rename(columns={"Feature": "Input"}), height=(len(vif) + 1) * 35 + 3,
              column_config={"VIF": st.column_config.NumberColumn(format="%.2f")})
    right.markdown(
        "Variance inflation factors (VIF) measure how much each input overlaps with the others. "
        "Values above about 5 would be a concern.\n\n"
        f"**The highest is {vif['VIF'].max():.2f}**, so no input is a near copy of the others and the "
        "coefficients can be read on their own."
    )

    section("2. Compare models with 5-fold cross-validation",
            "20% of movies are held out for a final test. Models are compared with 5-fold CV on "
            "the other 80%. Lower RMSE is better.")
    st.markdown(
        "- **Baseline:** predicts the average rating for every movie.\n"
        "- **No era:** popularity within the era, runtime, Oscar nomination, and genres.\n"
        "- **Era:** the same inputs plus the era.\n"
        "- **Era x features:** also lets each input's effect differ by era (interaction terms)."
    )
    cv = results["cv"]
    fig = px.bar(cv[cv["Estimator"].isin(["Mean", "OLS"])], x="Model", y="CV RMSE",
                 title="Cross-validated RMSE (OLS)", text_auto=".3f")
    fig.update_traces(marker_color=ERA_COLORS["Middle"])
    fig.update_yaxes(range=[0.8, 1.2])
    st.plotly_chart(style(fig, "", "CV RMSE"), width="stretch")
    wide = cv.pivot(index="Model", columns="Estimator", values="CV RMSE")
    wide.insert(0, "Inputs", cv.groupby("Model")["Inputs"].first())
    wide = wide.loc[["Baseline", "No era", "Era", "Era x features"], ["Inputs", "OLS", "Ridge", "Lasso"]]
    wide.loc["Baseline", "OLS"] = cv.loc[cv["Model"] == "Baseline", "CV RMSE"].iloc[0]  # predicts the mean
    table(wide.reset_index(), column_config={
        name: st.column_config.NumberColumn(format="%.4f") for name in ["OLS", "Ridge", "Lasso"]})
    st.caption("CV RMSE for each model and estimator (the baseline just predicts the average). Ridge "
               "and lasso match OLS to four decimals: with about 39,000 training movies and at most 47 "
               "inputs, there is no overfitting for the penalty to fix, so the final model is plain "
               "OLS. See Model checks.")

    section("3. The era gap",
            "From the Era model: how much higher old and middle-era movies rate than new movies with "
            "the same popularity rank, runtime, Oscar status, and genres. New movies are the "
            "reference group, so their gap is 0 by definition and the other eras are measured "
            "against them.")
    reference = pd.DataFrame([{"Era": "New (2000+)", "Rating gap vs. new movies": 0.0,
                               "Robust 95% CI": "reference group"}])
    table(pd.concat([results["era_gap"], reference], ignore_index=True), column_config={
        "Rating gap vs. new movies": st.column_config.NumberColumn(format="%+.3f"),
        "95% CI low": st.column_config.NumberColumn(format="%.3f"),
        "95% CI high": st.column_config.NumberColumn(format="%.3f"),
    })


def show_effects(results):
    section("What makes a movie good in each era?",
            "The same model fit separately for each era. Each dot is how much one input changes the "
            "rating in that era, with its 95% confidence interval. Dots that line up mean the input "
            "mattered the same way in every era; dots that spread apart mean the era changed what "
            "matters.")
    coefficients = results["coefficients"].assign(
        label=lambda df: df["feature"].map(lambda f: FEATURE_LABELS.get(f, f)))
    order = coefficients[coefficients["era"] == "New"].sort_values("coefficient")["label"].tolist()
    positions = {label: i for i, label in enumerate(order)}
    offsets = {"Old": 0.22, "Middle": 0, "New": -0.22}  # one row per input, eras side by side
    fig = go.Figure()
    for era in ERAS:
        rows = coefficients[coefficients["era"] == era]
        fig.add_trace(go.Scatter(
            x=rows["coefficient"], y=rows["label"].map(positions) + offsets[era], mode="markers",
            name=ERA_LABELS[era], marker={"color": ERA_COLORS[era], "size": 9},
            customdata=rows["label"],
            hovertemplate="%{customdata}: %{x:+.2f}<extra>" + ERA_LABELS[era] + "</extra>",
            error_x={"type": "data", "symmetric": False, "array": rows["high"] - rows["coefficient"],
                     "arrayminus": rows["coefficient"] - rows["low"], "color": ERA_COLORS[era],
                     "thickness": 1.5, "width": 0},
        ))
    fig.add_vline(x=0, line_color="#b5b4ae", line_width=1)
    fig.update_layout(title="Effect of each input on IMDb rating, by era (95% CI)", height=680,
                      legend={"orientation": "h", "y": -0.1})
    fig.update_yaxes(tickvals=list(positions.values()), ticktext=order, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    st.plotly_chart(style(fig, "Change in IMDb rating", ""), width="stretch")

    wide = coefficients.pivot(index="label", columns="era", values="coefficient")[ERAS].loc[order[::-1]]
    wide["Change, old to new"] = wide["New"] - wide["Old"]
    table(wide.reset_index().rename(columns={"label": "Input", **ERA_LABELS}).round(2),
          height=(len(wide) + 1) * 35 + 3)


def show_checks(results):
    holdout = results["holdout"]
    section("1. Final test on the holdout set",
            "The Era x features model, refit on all training movies and scored once on the 20% held "
            "out at the start.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline RMSE", f"{holdout['baseline_rmse']:.3f}")
    col2.metric("Model RMSE", f"{holdout['rmse']:.3f}")
    col3.metric("Model R-squared", f"{holdout['r2']:.3f}")

    section("2. Is the improvement bigger than fold-to-fold noise?",
            "CV RMSE for each of the 5 folds. If a model wins in every fold, the gain isn't luck.")
    folds = results["folds"].copy()
    folds.loc["Mean"] = folds.mean()
    folds.loc["Standard deviation"] = folds.iloc[:5].std()
    table(folds.reset_index().rename(columns={"index": "Fold"}).round(4))

    penalties = results["penalties"]
    section("3. Why ridge and lasso match OLS")
    st.markdown(
        f"- Ridge picked a penalty of **{penalties['ridge_alpha']:.2f}**, and lasso picked "
        f"**{penalties['lasso_alpha']:.5f}**.\n"
        f"- Lasso set **{penalties['lasso_zeroed']} of {penalties['inputs']}** coefficients to zero.\n"
        f"- With {penalties['train_movies']:,} training movies, the penalties barely change the fit."
    )

    assumptions = results["assumptions"]
    section("4. OLS assumptions",
            "Residuals should have the same spread everywhere and look roughly normal.")
    st.markdown(
        f"- **Breusch-Pagan p-value: {assumptions['bp_p']:.2g}.** The spread isn't constant, so the "
        "era gap was re-estimated with robust (HC3) standard errors. The confidence interval barely "
        "changes (see the era gap table on the Models section).\n"
        f"- **Residual skew: {assumptions['skew']:.2f}.** A few movies rate far below what the model "
        "predicts. With this many movies the coefficient estimates are still reliable."
    )
    left, right = st.columns(2)
    sample = np.random.default_rng(0).choice(len(holdout["residuals"]), size=4000, replace=False)
    fig = px.scatter(x=holdout["predicted"][sample], y=holdout["residuals"][sample], opacity=0.3,
                     title="Residuals vs. predicted (4,000 test movies)")
    fig.update_traces(marker={"color": ERA_COLORS["Middle"], "size": 4})
    fig.add_hline(y=0, line_color="#b5b4ae")
    left.plotly_chart(style(fig, "Predicted rating", "Residual"), width="stretch")
    quantiles = np.sort(holdout["residuals"])
    theoretical = sm.ProbPlot(quantiles).theoretical_quantiles
    fig = px.scatter(x=theoretical, y=quantiles, title="Normal Q-Q plot of residuals")
    fig.update_traces(marker={"color": ERA_COLORS["Middle"], "size": 3})
    fig.add_trace(go.Scatter(x=theoretical, y=theoretical * quantiles.std() + quantiles.mean(),
                             mode="lines", line={"color": "#b5b4ae"}, showlegend=False))
    right.plotly_chart(style(fig, "Theoretical quantiles", "Residual quantiles"), width="stretch")

    section("5. Do the results depend on where the eras are cut?",
            "The same Era model with different cutoffs between Old, Middle, and New.")
    table(results["cutoffs"].round(3))

    section("Limitations")
    st.markdown(
        "- **Survivorship:** old movies with 1,000+ votes are already the ones people chose to watch "
        "decades later.\n"
        "- **Today's raters:** IMDb ratings mostly come from people watching now, not audiences at "
        "release.\n"
        "- **\"Better\" means IMDb rating.** Budget and critic scores aren't in this model yet."
    )


def show_try_it():
    section("Predict a movie's rating in each era",
            "Describe a movie and the Era x features model predicts its IMDb rating as if it came "
            "out in each era.")
    model, genre_columns = prediction_model()
    left, right = st.columns([1, 1.3])
    with left:
        genres = st.multiselect("Genres (up to 3)", genre_columns, default=["Drama"], max_selections=3)
        runtime = st.slider("Runtime (minutes)", 60, 240, 110, step=5)
        popularity = st.slider("Popularity within its era (percentile)", 1, 100, 50,
                               help="100 = among the most-voted movies of its era")
        oscar = st.checkbox("Nominated for an Oscar")

    movie = {"popularity_pct": popularity / 100, "runtime_hours": runtime / 60,
             "oscar_nominated": int(oscar), **{genre: int(genre in genres) for genre in genre_columns}}
    rows = pd.DataFrame([{**movie, "era": era} for era in ERAS])
    rows["era"] = pd.Categorical(rows["era"], categories=ERAS)
    rows["Predicted rating"] = model.predict(rows)
    rows["Era"] = rows["era"].map(ERA_LABELS)
    with right:
        fig = px.bar(rows, x="Era", y="Predicted rating", text_auto=".2f",
                     color="era", color_discrete_map=ERA_COLORS, title="Predicted IMDb rating")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[0, 10])
        st.plotly_chart(style(fig, "", "Predicted IMDb rating"), width="stretch")
    st.caption("Typical prediction error is about 0.94 rating points (the holdout RMSE).")


def render():
    page_header("Era comparison: are older movies better?",
                "Comparing old, middle, and new movies on fair terms, then modeling what drives "
                "ratings in each era.")
    if missing_files(FILES, "Run `notebooks/movie_dataset.ipynb` and then "
                            "`notebooks/movie_data_wrangling.ipynb` to create them."):
        return
    movies, _ = load_movies()
    choice = subpages(["Overview", "Comparing the eras", "Models", "Effects by era", "Model checks",
                       "Try it"], key="era_section")
    if choice == "Overview":
        show_overview(movies)
    elif choice == "Comparing the eras":
        show_comparison()
    elif choice == "Models":
        show_models(model_results())
    elif choice == "Effects by era":
        show_effects(model_results())
    elif choice == "Model checks":
        show_checks(model_results())
    else:
        show_try_it()
