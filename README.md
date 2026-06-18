# World Cup Match Predictor

A machine learning model that predicts the outcome of international football
matches, built as the foundation for a 2026 World Cup tournament simulator.

This script (`world_cup_step1.py`) is **Part 1** of the project: it predicts a
single match between two teams as a set of probabilities. A later part will use
this to simulate the whole tournament and estimate each team's chance of winning.

## What it does

Given two teams, the model outputs three probabilities:

```
[ P(team1 wins),  P(draw),  P(team2 wins) ]
```

For example:

```
predict_match("Argentina", "Austria")  ->  [0.612, 0.232, 0.156]
```

## How it works

The model is trained on ~50,000 historical international matches. The approach
has three pieces:

1. **Label** — each past match is labelled as a home win (`0`), draw (`1`), or
   away win (`2`), based on the final score.
2. **Elo ratings** — every team gets a single strength number, built up match by
   match through history. Beating a strong team raises a team's rating more than
   beating a weak one. Each team's rating *before* a match is used as a feature.
3. **Classifier** — a model learns to map `[team1 Elo, team2 Elo, venue]` to the
   three outcome probabilities.

### Features

| Feature     | Meaning                                              |
|-------------|------------------------------------------------------|
| `home_elo`  | Strength rating of team 1 before the match           |
| `away_elo`  | Strength rating of team 2 before the match           |
| `is_home`   | `1` if team 1 played at home, `0` if neutral venue   |

### Models

Two models are trained and compared:

- **Logistic regression** — interpretable baseline.
- **Random forest** — ensemble method.

They are evaluated on held-out matches using accuracy and log loss. Log loss is
the more important metric here, because it rewards well-calibrated
*probabilities* (which matter for the eventual simulation), not just correct
guesses. The script uses logistic regression as the final model.

Typical performance (3-class problem, draws are inherently hard to predict):

```
Logistic regression   accuracy ~0.58   log_loss ~0.90
```

## Requirements

- Python 3
- pandas
- numpy
- scikit-learn

Install with:

```bash
pip install pandas numpy scikit-learn
```

## Data

Historical match results from the Kaggle dataset
[International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017).
Only `results.csv` is needed for this script. The file includes the upcoming
2026 World Cup fixtures (with empty scores), which are dropped during training
and reused later for the simulation.

## How to run

1. Download `results.csv` from the dataset above.
2. Open `match-predictor.py` and set `DATA_PATH` to the file's location on your
   computer:

   ```python
   DATA_PATH = "/path/to/results.csv"
   ```

3. Run it:

   ```bash
   python world_cup_step1.py
   ```

You should see the two models' scores printed, followed by example predictions.

## Using the predictor

```python
# Neutral venue (the default) — no home advantage for either team
predict_match("Brazil", "Argentina")

# Give team 1 a home advantage
predict_match("United States", "Brazil", neutral=False)
```

The output order follows the team order you pass in:
`[P(team1 wins), P(draw), P(team2 wins)]`.

**Note:** team names must match the dataset's spelling exactly, or the team
falls back to a default rating of 1500. For example, use `"United States"`, not
`"USA"`.

## Next steps

- Build the tournament simulator: play the 2026 bracket thousands of times using
  `predict_match`, then count how often each team wins to get championship odds.
- Optionally add more features (recent form, goal difference) and a host-nation
  home advantage.
