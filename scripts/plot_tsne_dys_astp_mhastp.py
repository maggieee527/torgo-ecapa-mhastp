import kaldiio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


def load_utt2spk(path):
    utt2spk = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
    return utt2spk


def load_embeddings(scp_path, utt2spk):
    utts = []
    embs = []
    labels = []

    for utt, emb in kaldiio.load_scp_sequential(scp_path):
        if utt not in utt2spk:
            continue
        utts.append(utt)
        embs.append(emb)
        labels.append(utt2spk[utt])

    X = np.stack(embs)
    X = normalize(X)  # speaker embedding 一般先做 L2 normalize 再可视化
    return utts, X, labels


def run_tsne(X):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    return tsne.fit_transform(X)


def plot_one(ax, X_2d, labels, title):
    spks = sorted(set(labels))

    for spk in spks:
        idx = [i for i, s in enumerate(labels) if s == spk]
        ax.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=12,
            alpha=0.8,
            label=spk,
        )

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


utt2spk_path = "data/dys_eval/utt2spk"
astp_scp = "exp/dys_astp/embeddings/dys_eval/xvector.scp"
mhastp_scp = "exp/dys_mhastp/embeddings/dys_eval/xvector.scp"

utt2spk = load_utt2spk(utt2spk_path)

_, X_astp, labels_astp = load_embeddings(astp_scp, utt2spk)
_, X_mhastp, labels_mhastp = load_embeddings(mhastp_scp, utt2spk)

print("ASTP embedding shape:", X_astp.shape)
print("MHASTP embedding shape:", X_mhastp.shape)

X_astp_2d = run_tsne(X_astp)
X_mhastp_2d = run_tsne(X_mhastp)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

plot_one(axes[0], X_astp_2d, labels_astp, "ECAPA-TDNN + ASTP")
plot_one(axes[1], X_mhastp_2d, labels_mhastp, "ECAPA-TDNN + MHASTP")

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Dysarthric Speaker ID",
    loc="upper center",
    ncol=8,
    fontsize=8,
    title_fontsize=9,
    frameon=False,
)

fig.suptitle("t-SNE visualization on TORGO dysarthric subset", fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.88])

out_path = "figures/tsne_dys_astp_vs_mhastp.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("saved:", out_path)
