"""Gym member segmentation: two segments on one life-stage axis.

Reads data/data.csv (2,000 members, seven fields) and writes every figure used on
the case-study page to one JSON file.

Usage:  python3 deep_dive.py data site_data.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
FIELDS = ['Gender', 'Marital Status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement Size']
LABELS = {
    'Gender': {0: 'Male', 1: 'Female'},
    'Marital Status': {0: 'Single', 1: 'Non-single'},
    'Education': {0: 'Other / unknown', 1: 'High school', 2: 'University', 3: 'Graduate school'},
    'Occupation': {0: 'Unemployed / unskilled', 1: 'Skilled employee / official', 2: 'Management / self-employed'},
    'Settlement Size': {0: 'Small city', 1: 'Mid-sized city', 2: 'Big city'},
}

# @load
df = pd.read_csv(DATA / 'data.csv')
out = {'totals': {'members': len(df), 'fields': len(FIELDS),
                  'missing': int(df.isna().sum().sum()), 'duplicates': int(df.duplicated().sum())}}

# @eda
# Counts and shares, not modes: every category with its proportion
out['eda'] = {c: [{'k': LABELS[c][v], 'n': int(n), 'share': float(100 * n / len(df))}
                  for v, n in df[c].value_counts().sort_index().items()] for c in LABELS}
out['numeric'] = {c: {'mean': float(df[c].mean()), 'median': float(df[c].median()), 'sd': float(df[c].std()),
                      'min': float(df[c].min()), 'max': float(df[c].max()), 'skew': float(df[c].skew())}
                  for c in ['Age', 'Income']}
out['spearman'] = {'fields': FIELDS, 'r': df[FIELDS].corr(method='spearman').round(3).values.tolist()}

# @income_age
# Income is almost an exact straight line in age
slope, intercept = np.polyfit(df['Age'], df['Income'], 1)
resid = df['Income'] - (intercept + slope * df['Age'])
r = df['Age'].corr(df['Income'])
out['income_age'] = {'slope': float(slope), 'intercept': float(intercept), 'r': float(r), 'r2': float(r * r),
                     'resid_sd': float(resid.std()), 'resid_max': float(resid.abs().max())}

# @brief_method
# The brief: standardise Age and Income only, then cluster on all seven fields
X = df[FIELDS].astype(float)
X[['Age', 'Income']] = StandardScaler().fit_transform(X[['Age', 'Income']])
inertia, sil = [], []
for k in range(1, 11):
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0).fit(X)
    inertia.append(float(model.inertia_))
    if k > 1:
        sil.append({'k': k, 'score': float(silhouette_score(X, model.labels_))})
out['elbow'] = [{'k': k + 1, 'inertia': v} for k, v in enumerate(inertia)]
out['silhouette'] = sil

# @segments
km = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0).fit_predict(X)
ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit_predict(X)
young = int(df.groupby(km)['Age'].mean().idxmin())          # label numbers are arbitrary:
ward_young = int(df.groupby(ward)['Age'].mean().idxmin())   # name each method's younger cluster
df['segment'] = np.where(km == young, 'Starters', 'Established')
df['silhouette'] = silhouette_samples(X, km)


def profile(g):
    return {
        'n': len(g), 'share': 100 * len(g) / len(df),
        'female': 100 * g['Gender'].mean(), 'non_single': 100 * g['Marital Status'].mean(),
        'age_mean': g['Age'].mean(), 'age_p10': g['Age'].quantile(0.1), 'age_p90': g['Age'].quantile(0.9),
        'income_median': g['Income'].median(), 'income_p10': g['Income'].quantile(0.1),
        'income_p90': g['Income'].quantile(0.9),
        'education': {LABELS['Education'][v]: 100 * (g['Education'] == v).mean() for v in range(4)},
        'occupation': {LABELS['Occupation'][v]: 100 * (g['Occupation'] == v).mean() for v in range(3)},
        'city': {LABELS['Settlement Size'][v]: 100 * (g['Settlement Size'] == v).mean() for v in range(3)},
        'age_bands': {b: 100 * m for b, m in pd.cut(g['Age'], [19, 29, 39, 49, 59, 99],
                                                    labels=['20-29', '30-39', '40-49', '50-59', '60+'])
                      .value_counts(normalize=True, sort=False).items()},
        'over_50': int((g['Age'] >= 50).sum()), 'over_60': int((g['Age'] >= 60).sum()),
        'under_30': int((g['Age'] < 30).sum()),
        'silhouette': g['silhouette'].mean(),
    }


out['segments'] = {s: profile(df[df.segment == s]) for s in ['Starters', 'Established']}
out['ward_check'] = {'sizes': np.bincount(ward).tolist(),
                     'age': df.groupby(ward)['Age'].mean().round(2).tolist(),
                     'income': df.groupby(ward)['Income'].mean().round(2).tolist()}
out['overlap'] = pd.crosstab(df.segment, np.where(ward == ward_young, 'Starters', 'Established')) \
    .reindex(index=['Starters', 'Established'], columns=['Starters', 'Established']).values.tolist()

# @agreement
# The same split under other seeds, linkages, encodings and distances


def agree(labels, X_=None, metric='euclidean'):
    same = max((labels == km).mean(), (labels != km).mean())
    s = silhouette_score(X_, labels, metric=metric) if X_ is not None else None
    return {'same': float(100 * same), 'ari': float(adjusted_rand_score(km, labels)), 'silhouette': s}


def kmeans2(M):
    return KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0).fit_predict(M)


seeds = [adjusted_rand_score(km, KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=s).fit_predict(X))
         for s in range(30)]
Z = StandardScaler().fit_transform(df[FIELDS].astype(float))
X5 = X[['Age', 'Income', 'Education', 'Occupation', 'Settlement Size']]
num, cat = ['Age', 'Income'], ['Gender', 'Marital Status', 'Education', 'Occupation', 'Settlement Size']
gower = sum(np.abs(df[c].values[:, None] - df[c].values[None, :]) / np.ptp(df[c].values) for c in num)
gower = (gower + sum((df[c].values[:, None] != df[c].values[None, :]).astype(float) for c in cat)) / len(FIELDS)
out['agreement'] = [
    {'k': 'K-means++, 30 random starts', 'group': 'seeds', 'same': 100.0, 'ari': float(min(seeds)), 'silhouette': None},
    dict(k='Agglomerative, Ward linkage', group='linkage', **agree(ward, X)),
    dict(k='Agglomerative, average linkage', group='linkage',
         **agree(AgglomerativeClustering(n_clusters=2, linkage='average').fit_predict(X), X)),
    dict(k='Agglomerative, complete linkage', group='linkage',
         **agree(AgglomerativeClustering(n_clusters=2, linkage='complete').fit_predict(X), X)),
    dict(k='All seven fields standardised', group='encoding', **agree(kmeans2(Z), Z)),
    dict(k='Without income', group='encoding', **agree(kmeans2(X.drop(columns='Income')), X.drop(columns='Income'))),
    dict(k='Without gender and marital status', group='encoding', **agree(kmeans2(X5), X5)),
    dict(k='Gower distance, average linkage', group='distance',
         **agree(AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average').fit_predict(gower),
                 gower, 'precomputed')),
]

# @single_fields
# How well one field on its own reproduces the two segments
is_est = (df.segment == 'Established').astype(int)


def best_rule(col):
    if col in ('Age', 'Income'):
        cuts = np.unique(df[col])
        acc, cut = max(((df[col] > c).astype(int).eq(is_est).mean(), c) for c in cuts)
        return float(100 * acc), float(cut)
    t = pd.crosstab(df[col], is_est)
    return float(100 * t.max(axis=1).sum() / len(df)), None


out['single_fields'] = [{'k': c, 'accuracy': best_rule(c)[0], 'cut': best_rule(c)[1]} for c in FIELDS]
rule = (df['Settlement Size'].eq(2) | df['Occupation'].ge(1)).astype(int)
out['rule_city_occupation'] = float(100 * rule.eq(is_est).mean())

# @city
out['city'] = [{'k': LABELS['Settlement Size'][v],
                'starters': int(((df['Settlement Size'] == v) & (df.segment == 'Starters')).sum()),
                'established': int(((df['Settlement Size'] == v) & (df.segment == 'Established')).sum())}
               for v in [2, 1, 0]]
small_est = df[(df['Settlement Size'] == 0) & (df.segment == 'Established')]
out['small_city_established'] = {'n': len(small_est), 'age_mean': float(small_est['Age'].mean()),
                                 'income_median': float(small_est['Income'].median()),
                                 'skilled_or_mgmt': float(100 * small_est['Occupation'].ge(1).mean()),
                                 'university_plus': float(100 * small_est['Education'].ge(2).mean())}

# @borderline
# Members the two-segment split fits worst
border = df[df.silhouette < 0.25]
out['borderline'] = {'n': len(border), 'share': float(100 * len(border) / len(df)),
                     'small_city': float(100 * border['Settlement Size'].eq(0).mean()),
                     'skilled_or_mgmt': float(100 * border['Occupation'].ge(1).mean()),
                     'university_plus': float(100 * border['Education'].ge(2).mean()),
                     'non_single': float(100 * border['Marital Status'].mean()),
                     'age_mean': float(border['Age'].mean()), 'age_min': int(border['Age'].min()),
                     'age_max': int(border['Age'].max()),
                     'small_city_established': int(((border['Settlement Size'] == 0)
                                                    & (border.segment == 'Established')).sum())}
out['silhouette_hist'] = np.histogram(df.silhouette, bins=np.arange(-0.1, 0.81, 0.05))[0].tolist()

# @points
# Every member for the scatter: age, income in $000s, segment, borderline flag
out['points'] = [[int(a), round(i / 1000, 1), int(s == 'Established'), int(b < 0.25)]
                 for a, i, s, b in zip(df['Age'], df['Income'], df.segment, df.silhouette)]

OUT.write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
print('wrote', OUT)
