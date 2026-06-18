import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# ---------------------------------------------------------------------------
# CHANGE THIS to wherever results.csv lives on your computer:
DATA_PATH = "/Users/areeb/Desktop/Projects-Machine-Learning/data/results.csv"
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["home_score", "away_score"]).copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# The thing we want to predict: the match outcome, from the home team's view.
#   0 = home team won
#   1 = draw
#   2 = away team won
def outcome(row):
    if row.home_score > row.away_score:
        return 0
    elif row.home_score == row.away_score:
        return 1
    else:
        return 2

df["result"] = df.apply(outcome, axis=1)


# Elo is a single number for how strong a team is. Everyone starts at 1500.
# After each match the winner takes points from the loser; beating a strong
# team earns more than beating a weak one. We walk through every match in
# time order and, crucially, record each team's rating before the match —
# that's what the model is allowed to "know" when predicting.

ratings = {}            # team name -> current Elo rating
START = 1500.0          # everyone starts here
K = 30.0                # how fast ratings move after each game

home_elo_before = np.zeros(len(df))
away_elo_before = np.zeros(len(df))

for i, row in enumerate(df.itertuples()):
    rh = ratings.get(row.home_team, START)
    ra = ratings.get(row.away_team, START)

    # Record the ratings as they stand BEFORE this match (these become features)
    home_elo_before[i] = rh
    away_elo_before[i] = ra

    # How likely was the home team to win, according to current ratings?
    expected_home = 1 / (1 + 10 ** ((ra - rh) / 400))

    # What actually happened (1 = home won, 0.5 = draw, 0 = home lost)
    actual_home = 1.0 if row.result == 0 else (0.5 if row.result == 1 else 0.0)

    # Update both teams. If home did better than expected, its rating rises.
    ratings[row.home_team] = rh + K * (actual_home - expected_home)
    ratings[row.away_team] = ra + K * ((1 - actual_home) - (1 - expected_home))

df["home_elo"] = home_elo_before
df["away_elo"] = away_elo_before

# Venue feature: 1 if the home team genuinely played at home, 0 if neutral.
df["is_home"] = (~df["neutral"]).astype(int)



# Football from the 1800s won't help predict 2026, so train on modern games.
train = df[df["date"] >= "2000-01-01"]

FEATURES = ["home_elo", "away_elo", "is_home"]
X = train[FEATURES]
y = train["result"]

# Hold out 20% of matches to test on (the model never sees these while training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
forest = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

for name, model in [("Logistic regression", logreg), ("Random forest", forest)]:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    print(f"{name:20s}  accuracy={acc:.3f}  log_loss={ll:.3f}")

# Lower log_loss = better-calibrated probabilities, which is what matters most
# for a simulation. Pick the model with the lower log_loss here.
best_model = logreg

def predict_match(team1,team2, neutral=True, model=best_model):
    """Returns [P(team 1 wins), P(draw), P(team 2 wins)] for a match."""
    rh = ratings.get(team1, START)
    ra = ratings.get(team2, START)
    row = pd.DataFrame([[rh, ra, 0 if neutral else 1]], columns=FEATURES)
    return model.predict_proba(row)[0]

print()
#choose what game you want to predict the result of

print("Argentina vs Austria: ", np.round(predict_match("Argentina", "Austria"), 3))
