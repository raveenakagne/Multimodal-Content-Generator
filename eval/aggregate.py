# eval/aggregate.py
import pandas as pd, pathlib, argparse
def main(run):
    scores = pd.read_csv(f"runs/{run}/scores.csv")
    agg = scores.groupby(["model","metric"]).score.agg(["mean","sem"]).reset_index()
    out = pathlib.Path(f"runs/{run}/agg.csv"); agg.to_csv(out, index=False)
    print(f"✓ aggregated metrics saved to {out}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("run")
    main(parser.parse_args().run)
