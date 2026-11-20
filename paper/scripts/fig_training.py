"""Training-loss curves, all four folds on one axis."""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P
import matplotlib.pyplot as plt

# Water-year colours come from paper_style so a given year is the same
# colour in every figure that encodes years as series. Defining a local
# map here gave WY2003 purple in this figure and navy in the
# precipitation figure.
FC = {f: P.WY_COLOR[P.FOLD_WY[f]] for f in P.FOLDS}


def losses(fold):
    p = os.path.join(P.DDPM, "Results", f"groundwater_f15_200ep_fold{fold}",
                     "logs", "metrics.csv")
    rows = list(csv.DictReader(open(p)))
    return ([int(r["epoch"]) for r in rows],
            [float(r["avg_loss"]) for r in rows])


def main():
    P.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    fin = {}
    for f in P.FOLDS:
        ep, tr = losses(f)
        fin[f] = tr[-1]
        ax.plot(ep, tr, color=FC[f], lw=P.LW_MAIN,
                label=f"WY{P.FOLD_WY[f]} held out")
    ax.set_title("Full schedule", fontsize=P.FS_TITLE)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss [-]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    # The final losses are no longer boxed onto the panel: they are four
    # numbers the text can state, and the box competed with the curves it sat
    # among. They are still printed to stdout below, and the curves themselves
    # show convergence, which is what the figure is for.
    P.save(fig, "fig_training_loss")
    for f in P.FOLDS:
        print(f"   WY{P.FOLD_WY[f]} final loss {fin[f]:.4f}")


if __name__ == "__main__":
    main()
