"""
TP - Mineração de Dados: Clusterização (K-Means x Hierárquico x GMM x DBSCAN)
Base: Customer Personality Analysis (Kaggle) -> arquivo marketing_campaign.csv (separador TAB)

Como usar:
    1) Baixe o CSV do Kaggle e coloque na mesma pasta deste script (ou ajuste CSV_PATH).
    2) pip install pandas numpy scikit-learn matplotlib seaborn
    3) python clusterizacao_customer_personality.py
Saídas (tabelas CSV e figuras PNG) vão para a pasta ./outputs
"""
import os
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
CSV_PATH = "marketing_campaign.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# 1. CARGA E DIAGNÓSTICO INICIAL
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH, sep="\t")
print("Shape original:", df.shape)

diag = pd.DataFrame({
    "tipo": df.dtypes.astype(str),
    "ausentes": df.isna().sum(),
    "%ausentes": (df.isna().mean() * 100).round(2),
    "n_unicos": df.nunique(),
})
diag.to_csv(f"{OUT}/diagnostico_inicial.csv")
print(diag)
print(df.describe().T)

# ---------------------------------------------------------------------------
# 2. ENGENHARIA DE ATRIBUTOS E LIMPEZA POR REGRAS DE DOMÍNIO
#    (regras fixas, não aprendidas dos dados -> não geram vazamento)
# ---------------------------------------------------------------------------
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
ref_date = df["Dt_Customer"].max()                 # data de referência da base
df["Age"] = ref_date.year - df["Year_Birth"]
df["Tenure_days"] = (ref_date - df["Dt_Customer"]).dt.days
df["Children"] = df["Kidhome"] + df["Teenhome"]
mnt_cols = [c for c in df.columns if c.startswith("Mnt")]
df["Total_Spent"] = df[mnt_cols].sum(axis=1)       # usado só no perfilamento

# categorias inconsistentes em Marital_Status -> binário "Partner"
partner = {"Married", "Together"}
df["Partner"] = df["Marital_Status"].apply(lambda x: 1 if x in partner else 0)

# Education -> ordinal (ordem natural de escolaridade)
edu_map = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
df["Education_ord"] = df["Education"].map(edu_map)

# registros impossíveis/errôneos (ex.: Age > 100, renda 666.666)
antes = len(df)
df = df[(df["Age"] <= 100) & ((df["Income"] < 600_000) | df["Income"].isna())].copy()
print(f"Removidos {antes - len(df)} registros por regra de domínio.")

features = (["Age", "Income", "Kidhome", "Teenhome", "Recency", "Tenure_days",
             "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
             "NumStorePurchases", "NumWebVisitsMonth", "Education_ord", "Partner"]
            + mnt_cols)
X_raw = df[features].reset_index(drop=True)
df = df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO SEM VAZAMENTO
#    Todo parâmetro aprendido (mediana, limites IQR, média/desvio) é ajustado
#    SOMENTE no conjunto de treino e apenas aplicado no holdout.
# ---------------------------------------------------------------------------
X_tr_raw, X_ho_raw = train_test_split(X_raw, test_size=0.2, random_state=RANDOM_STATE)

imputer = SimpleImputer(strategy="median").fit(X_tr_raw)
X_tr_imp = pd.DataFrame(imputer.transform(X_tr_raw), columns=features, index=X_tr_raw.index)
X_ho_imp = pd.DataFrame(imputer.transform(X_ho_raw), columns=features, index=X_ho_raw.index)

# winsorização por IQR (limites aprendidos no treino) apenas em variáveis contínuas
cont_cols = ["Age", "Income", "Tenure_days"] + mnt_cols + \
            ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
             "NumStorePurchases", "NumWebVisitsMonth"]
q1, q3 = X_tr_imp[cont_cols].quantile(0.25), X_tr_imp[cont_cols].quantile(0.75)
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
X_tr_imp[cont_cols] = X_tr_imp[cont_cols].clip(low, high, axis=1)
X_ho_imp[cont_cols] = X_ho_imp[cont_cols].clip(low, high, axis=1)

scaler = StandardScaler().fit(X_tr_imp)
X_tr = scaler.transform(X_tr_imp)
X_ho = scaler.transform(X_ho_imp)
print("Treino:", X_tr.shape, "| Holdout:", X_ho.shape)

# Visualização: boxplots antes/depois do tratamento de outliers (treino)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(data=X_tr_raw[["Income", "Age", "Tenure_days"]].pipe(
    lambda d: (d - d.mean()) / d.std()), ax=ax[0])
ax[0].set_title("Antes (padronizado, com outliers)")
sns.boxplot(data=pd.DataFrame(X_tr, columns=features)[["Income", "Age", "Tenure_days"]], ax=ax[1])
ax[1].set_title("Depois (imputado, winsorizado, padronizado)")
plt.tight_layout(); plt.savefig(f"{OUT}/outliers_antes_depois.png", dpi=150); plt.close()

# Matriz de correlação
plt.figure(figsize=(11, 9))
sns.heatmap(pd.DataFrame(X_tr, columns=features).corr(), cmap="coolwarm", center=0)
plt.title("Correlação entre atributos (treino)")
plt.tight_layout(); plt.savefig(f"{OUT}/correlacao.png", dpi=150); plt.close()

# PCA 2D só para visualização
pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_tr)
Z_tr = pca.transform(X_tr)
print("Variância explicada PCA(2):", pca.explained_variance_ratio_.round(3))

# ---------------------------------------------------------------------------
# 4. FUNÇÕES DE AVALIAÇÃO
#    Métricas internas (sem rótulo verdadeiro) + estabilidade (ARI por subamostragem),
#    que substitui a validação cruzada em aprendizado não supervisionado.
# ---------------------------------------------------------------------------
def metricas_internas(X, labels):
    mask = labels != -1                              # ignora ruído do DBSCAN
    if len(set(labels[mask])) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(
        silhouette=silhouette_score(X[mask], labels[mask]),
        davies_bouldin=davies_bouldin_score(X[mask], labels[mask]),
        calinski_harabasz=calinski_harabasz_score(X[mask], labels[mask]),
    )

def estabilidade(make_model, X, n_rep=10, frac=0.8):
    """ARI médio entre a partição completa e partições em subamostras de 80%."""
    full = make_model().fit_predict(X)
    rng = np.random.RandomState(RANDOM_STATE)
    aris = []
    for _ in range(n_rep):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        sub = make_model().fit_predict(X[idx])
        aris.append(adjusted_rand_score(full[idx], sub))
    return np.mean(aris), np.std(aris)

# ---------------------------------------------------------------------------
# 5. MODELAGEM COMPARATIVA E AJUSTE DE HIPERPARÂMETROS
# ---------------------------------------------------------------------------
resultados = []
K_RANGE = range(2, 11)

# 5.1 K-Means (n_clusters, init) + cotovelo
inercias = {}
for k, init in itertools.product(K_RANGE, ["k-means++", "random"]):
    km = KMeans(n_clusters=k, init=init, n_init=10, random_state=RANDOM_STATE).fit(X_tr)
    m = metricas_internas(X_tr, km.labels_)
    m_ho = metricas_internas(X_ho, km.predict(X_ho))          # generalização no holdout
    if init == "k-means++":
        inercias[k] = km.inertia_
    resultados.append(dict(algoritmo="KMeans", config=f"k={k}, init={init}", k=k,
                           silhouette_holdout=m_ho["silhouette"], **m))

# 5.2 Hierárquico aglomerativo (linkage x k)
for k, link in itertools.product(K_RANGE, ["ward", "average", "complete", "single"]):
    ag = AgglomerativeClustering(n_clusters=k, linkage=link).fit(X_tr)
    resultados.append(dict(algoritmo="Hierarquico", config=f"k={k}, linkage={link}", k=k,
                           **metricas_internas(X_tr, ag.labels_)))

# 5.3 Gaussian Mixture (componentes x covariância) + BIC
bics = {}
for k, cov in itertools.product(K_RANGE, ["full", "tied", "diag", "spherical"]):
    gm = GaussianMixture(n_components=k, covariance_type=cov, n_init=3,
                         random_state=RANDOM_STATE).fit(X_tr)
    lab = gm.predict(X_tr)
    bics[(k, cov)] = gm.bic(X_tr)
    resultados.append(dict(algoritmo="GMM", config=f"k={k}, cov={cov}", k=k, bic=gm.bic(X_tr),
                           silhouette_holdout=metricas_internas(X_ho, gm.predict(X_ho))["silhouette"],
                           **metricas_internas(X_tr, lab)))

# 5.4 DBSCAN (eps x min_samples) + gráfico de k-distância
nn = NearestNeighbors(n_neighbors=10).fit(X_tr)
kdist = np.sort(nn.kneighbors(X_tr)[0][:, -1])
plt.figure(figsize=(6, 4)); plt.plot(kdist)
plt.title("Gráfico de k-distância (k=10)"); plt.xlabel("Pontos ordenados"); plt.ylabel("Distância")
plt.tight_layout(); plt.savefig(f"{OUT}/dbscan_kdist.png", dpi=150); plt.close()

for eps, ms in itertools.product(np.arange(1.5, 5.1, 0.5), [5, 10, 15, 20]):
    db = DBSCAN(eps=eps, min_samples=ms).fit(X_tr)
    n_cl = len(set(db.labels_) - {-1})
    ruido = float(np.mean(db.labels_ == -1))
    resultados.append(dict(algoritmo="DBSCAN", config=f"eps={eps:.1f}, min_samples={ms}",
                           k=n_cl, pct_ruido=ruido, **metricas_internas(X_tr, db.labels_)))

res = pd.DataFrame(resultados)
res.to_csv(f"{OUT}/resultados_todas_configuracoes.csv", index=False)

# Gráficos de seleção
plt.figure(figsize=(6, 4)); plt.plot(list(inercias), list(inercias.values()), "o-")
plt.title("K-Means: método do cotovelo"); plt.xlabel("k"); plt.ylabel("Inércia")
plt.tight_layout(); plt.savefig(f"{OUT}/kmeans_cotovelo.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for alg in ["KMeans", "Hierarquico", "GMM"]:
    d = res[res.algoritmo == alg].groupby("k")["silhouette"].max()
    ax[0].plot(d.index, d.values, "o-", label=alg)
ax[0].set_title("Melhor silhouette por k"); ax[0].set_xlabel("k"); ax[0].legend()
for cov in ["full", "tied", "diag", "spherical"]:
    ax[1].plot(list(K_RANGE), [bics[(k, cov)] for k in K_RANGE], "o-", label=cov)
ax[1].set_title("GMM: BIC por covariância"); ax[1].set_xlabel("componentes"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/comparacao_k.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# 6. SELEÇÃO DA MELHOR CONFIG DE CADA ALGORITMO + ESTABILIDADE
#    Critério: maior silhouette (com k >= 3 para segmentação útil ao negócio;
#    DBSCAN exige ruído <= 30%). Ajuste o critério conforme sua discussão.
# ---------------------------------------------------------------------------
cand = res[res.k >= 3].copy()
cand = cand[~((cand.algoritmo == "DBSCAN") & (cand.pct_ruido > 0.30))]
melhores = cand.sort_values("silhouette", ascending=False).groupby("algoritmo").head(1)
print(melhores[["algoritmo", "config", "k", "silhouette", "davies_bouldin", "calinski_harabasz"]])

def construir(row):
    k, cfg = int(row.k), row.config
    if row.algoritmo == "KMeans":
        init = cfg.split("init=")[1]
        return lambda: KMeans(n_clusters=k, init=init, n_init=10, random_state=RANDOM_STATE)
    if row.algoritmo == "Hierarquico":
        link = cfg.split("linkage=")[1]
        return lambda: AgglomerativeClustering(n_clusters=k, linkage=link)
    if row.algoritmo == "GMM":
        cov = cfg.split("cov=")[1]
        return lambda: GaussianMixture(n_components=k, covariance_type=cov, n_init=3, random_state=RANDOM_STATE)
    eps = float(cfg.split("eps=")[1].split(",")[0]); ms = int(cfg.split("min_samples=")[1])
    return lambda: DBSCAN(eps=eps, min_samples=ms)

final = []
for _, row in melhores.iterrows():
    mk = construir(row)
    ari_m, ari_s = estabilidade(mk, X_tr)
    final.append(dict(algoritmo=row.algoritmo, config=row.config, k=row.k,
                      silhouette=row.silhouette, davies_bouldin=row.davies_bouldin,
                      calinski_harabasz=row.calinski_harabasz,
                      estabilidade_ARI_media=ari_m, estabilidade_ARI_dp=ari_s))
final = pd.DataFrame(final).sort_values("silhouette", ascending=False)
final.to_csv(f"{OUT}/comparacao_final.csv", index=False)
print(final.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. VISUALIZAÇÃO (PCA 2D) E PERFIL DE NEGÓCIO DOS CLUSTERS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(melhores), figsize=(5 * len(melhores), 4.5))
axes = np.atleast_1d(axes)
labels_final = {}
for ax, (_, row) in zip(axes, melhores.iterrows()):
    lab = construir(row)().fit_predict(X_tr)
    labels_final[row.algoritmo] = lab
    ax.scatter(Z_tr[:, 0], Z_tr[:, 1], c=lab, cmap="tab10", s=10)
    ax.set_title(f"{row.algoritmo}\n{row.config}")
plt.tight_layout(); plt.savefig(f"{OUT}/clusters_pca.png", dpi=150); plt.close()

# Perfil do melhor algoritmo (variáveis na escala original, treino)
melhor_alg = final.iloc[0].algoritmo
perfil_df = df.loc[X_tr_raw.index].copy()
perfil_df["cluster"] = labels_final[melhor_alg]
cols_perfil = ["Age", "Income", "Children", "Total_Spent", "Recency", "Tenure_days",
               "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
               "NumDealsPurchases", "NumWebVisitsMonth"] + mnt_cols
perfil = perfil_df.groupby("cluster")[cols_perfil].mean().round(1)
perfil["tamanho"] = perfil_df.groupby("cluster").size()
perfil["%"] = (perfil["tamanho"] / perfil["tamanho"].sum() * 100).round(1)
perfil.to_csv(f"{OUT}/perfil_clusters_{melhor_alg}.csv")
print(f"\nPerfil dos clusters ({melhor_alg}):\n", perfil.T)

plt.figure(figsize=(10, 5))
sns.heatmap((perfil[cols_perfil] - perfil[cols_perfil].mean()) / perfil[cols_perfil].std(),
            annot=perfil[cols_perfil], fmt=".0f", cmap="RdBu_r", center=0)
plt.title(f"Perfil dos clusters - {melhor_alg} (cor = desvio relativo; número = média)")
plt.tight_layout(); plt.savefig(f"{OUT}/perfil_heatmap.png", dpi=150); plt.close()

print("\nConcluído. Veja a pasta:", OUT)
