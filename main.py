"""
NBA Betting Model
─────────────────
SETUP (first time only):
  1. pip install -r requirements.txt
  2. Copy .env.example to .env  and add your free Odds API key
     → Sign up at https://the-odds-api.com/  (500 free requests/month)
  3. python main.py train      ← pulls 3 seasons of NBA data, trains models (~10-20 min)
  4. python main.py predict    ← shows tonight's picks

DAILY USE:
  python main.py predict       ← re-run each day to see today's picks
  python main.py train         ← retrain every few weeks to include recent games
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict"):
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "train":
        from models.train import train_all_models
        train_all_models()

    elif command == "predict":
        from models.predict import run_predictions
        upcoming = "--upcoming" in sys.argv
        run_predictions(upcoming=upcoming)


if __name__ == "__main__":
    main()
