# Gym Member Segmentation

[![Case Study](https://img.shields.io/badge/Case_Study-Read_the_page-111111?style=for-the-badge)](https://racheltran184.github.io/Python-Project-3/)
[![Python](https://img.shields.io/badge/Python-scikit--learn_·_pandas-3776AB?style=for-the-badge&logo=python&logoColor=white)](#how-to-run)

**Two kinds of member, one life stage apart.** K-means++ and agglomerative clustering on 2,000 gym members, then stress-tested six ways. Every method finds the same two segments, but they sit on one life-stage axis: where members live already sorts them, and none of it needs a member's gender or marital status.

![All 2,000 gym members as dots, stacked by age from 20 to 76: Starters in grey at the younger end, Established members in white at the older end, mixed between about 35 and 47](gym-segmentation.jpg)

*Every member as one dot, stacked by age: Starters in grey, Established members in white.*

---

## Overview

A large national gym chain wants to understand its members. The brief: find distinct segments in the membership details of 2,000 members, describe them for management, and recommend marketing for each segment, using k-means++ and agglomerative clustering on all seven fields. `Task.ipynb` and `Report.pdf` are the original project. The [case study](https://racheltran184.github.io/Python-Project-3/) takes it deeper with `deep_dive.py`: segments described in proportions, six stress tests, single-field rules and the members who sit between the segments.

| Item | Detail |
|---|---|
| **Project** | Individual project for a postgraduate predictive analytics unit (Semester 1, 2025) |
| **Audience** | Management at the gym chain, and its marketing, club and product teams |
| **Data** | 2,000 members, seven fields: gender, marital status, age, education, self-reported annual income in US dollars, occupation and city size |
| **Method** | K-means++ and agglomerative clustering (Ward linkage) on all seven fields, age and income standardised; k chosen by the elbow method and silhouette; no dimensionality reduction |
| **Stack** | Python, pandas, NumPy, scikit-learn, matplotlib, seaborn |

### Headline numbers

| Figure | What it means |
|---:|---|
| **2,000** | Members, seven fields, no missing values |
| **0.545** | Silhouette at two segments, against 0.433 at three and 0.452 at four |
| **97.9%** | Members that Ward linkage puts in the same segment as k-means++ |
| **99.4%** | Members who keep their segment when gender and marital status are left out |
| **r = 0.9995** | Income against age: one field repeats the other |

### What made the data tricky

- **One axis.** Income follows age almost exactly, and marital status, job, education and city move together. Seven fields carry about one dimension of information.
- **Mixed types.** Two yes-or-no fields, three coded categories and dollars share one distance scale. The brief standardises only age and income, so one step in education counts as much as nine and a half years of age.
- **Sensitive fields.** Two of the seven fields are gender and marital status, which members may not expect a gym to use for targeting.

---

## Method

| Stage | What it does |
|---|---|
| 1. Check | Missing values and duplicates (none and 3); income against age; rank correlations between all seven fields |
| 2. Scale | Z-scores for age and income, as the brief requires; the coded fields keep their codes |
| 3. Choose k | Inertia for k = 1 to 10 and silhouette for k = 2 to 10: inertia falls 62% from one segment to two, and silhouette peaks at two |
| 4. Cluster | K-means++ with 10 starts and a fixed seed; agglomerative clustering with Ward linkage; segments described in proportions |
| 5. Stress-test | Change one assumption at a time and count the members who keep their segment |

```python
X = df[FIELDS].astype(float)
X[['Age', 'Income']] = StandardScaler().fit_transform(X[['Age', 'Income']])
# ↑ the brief: standardise the two numeric fields only
km = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0).fit_predict(X)
ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit_predict(X)
df['silhouette'] = silhouette_samples(X, km)   # below 0.25: almost as close to the other segment
```

| Check against k-means++ on the brief's seven fields | Same segment | Adjusted Rand index |
|---|---:|---:|
| K-means++, 30 random starts | 100% | 1.000 |
| Agglomerative, Ward linkage | 97.9% | 0.918 |
| Agglomerative, average linkage | 98.8% | 0.953 |
| Agglomerative, complete linkage | 90.7% | 0.661 |
| All seven fields standardised | 98.1% | 0.925 |
| Without income | 98.7% | 0.949 |
| Without gender and marital status | 99.4% | 0.976 |
| Gower distance, average linkage | 98.0% | 0.920 |

*Same segment is the share of the 2,000 members that a check puts in the same segment as k-means++, after matching labels. The notebook's agglomerative cell first used complete linkage, while the report's table matched Ward, scikit-learn's default; it now uses Ward and a fixed k-means seed, so it reproduces the report's tables: k-means clusters of 1,024 and 976 members, Ward clusters of 1,008 and 992.*

---

## Six findings

*Segments: k-means++ on all seven fields with age and income standardised, k = 2, named by life stage. Seven membership fields and no behaviour: the data says who members are, not how often they come, what they pay or who leaves.*

**1. The data has one axis, not seven.** Income follows age almost exactly: income ≈ −$61,784 + $4,882 × age (r = 0.9995), so age alone predicts a member's income with a typical error of about $1,400. Marital status moves with job type (rank correlation 0.95), education (0.88) and city size (0.79). The clustering finds the two ends of one line: younger, single, school-educated members in unskilled work in smaller cities, and older, non-single, university-educated professionals in big cities.

**2. Two segments, and every method agrees.** Two segments score a silhouette of 0.545, against 0.433 for three and 0.452 for four, and thirty random starts return the same split every time. Ward linkage puts 97.9% of members in the same segment as k-means, average linkage 98.8% and Gower distance 98.0%. Only complete linkage, which judges clusters by their farthest members, drifts to 90.7%.

**3. Two profiles, in proportions.**

| | Starters | Established |
|---|---:|---:|
| Members | 1,024 (51%) | 976 (49%) |
| Mean age (most members aged) | 34 (27–41) | 48 (40–57) |
| Female | 36.5% | 85.6% |
| Non-single (married, separated, divorced or widowed) | 3.2% | 99.2% |
| University or graduate school | 2.6% | 92.6% |
| Skilled or management job | 3.8% | 99.3% |
| Lives in a big city | 0% | 81.8% |
| Aged 50 or over | 1 member | 403 (41.3%) |
| Median income | $102,356 | $171,798 |

*"Most members aged" is the 10th to 90th percentile.* The first report called Starters "young, single males", but 37% are women; and calling Established members "urban" misses the 178 who live in small cities.

**4. Where members live already sorts them.** All 798 big-city members are Established, and all 72 mid-sized-city members are Starters. Small cities hold both: 952 Starters (84%) and 178 Established members (16%). Those 178 look like big-city Established members in everything but address: mean age 49, median income $175,545, 97% in skilled or management work and 90% university-educated.

**5. No need to ask gender or marital status.** Clustering without gender and marital status keeps 99.4% of members in the same segment. Marital status alone matches the split for 98.0% of members, but job type alone matches 97.7%, and a two-part rule (big city, or a skilled or management job, means Established) matches 97.8%. Gender is the weakest guide: on its own it reproduces only 74% of the split.

**6. One member in fifteen sits in between.** 134 members (6.7%) score a silhouette below 0.25: they are almost as close to the other segment as to their own. Most are non-single (88%), in skilled or management work (88%) and university-educated (77%), aged 28–59, and three in four live in small cities. 73 of them are small-city Established members: by job and education they're Established, by address they're Starters.

---

## Actions: the next 0–6 months

| Team | Action | Why | Track |
|---|---|---|---|
| Offers & pricing | Lead with flexible plans for Starters: month-to-month and off-peak plans with no joining fee in small and mid-sized city clubs | Starters are 1,024 members (51%); 96% are unemployed or in unskilled work, and their median income is 40% below Established members' | 90-day retention of new Starters |
| Offers & pricing | Sell employer and premium plans to Established: corporate wellness agreements with big-city employers, and a premium tier with booked express sessions | 99% of Established members work in skilled or management roles, and 82% live in big cities | Members joining through employer agreements |
| Offers & pricing | Let the in-between members choose: show every new member the same three tiers (flex, standard and premium) | 134 members (6.7%) sit between the segments; 73 are Established members in small cities who would otherwise get Starter offers | Tier mix and upgrades by segment |
| Clubs & classes | Set each club's format by its city: express classes and a premium tier in big cities, flexible plans and community classes in small and mid-sized cities | All 798 big-city members are Established; all 72 mid-sized-city members and 952 of 1,130 small-city members are Starters | Joins and 90-day retention by club format |
| Clubs & classes | Keep a premium option and an evening class block in small-city clubs | 178 Established members (16% of small-city members) live there: mean age 49, 97% in skilled or management work | Premium take-up in small-city clubs |
| Clubs & classes | Programme for members over 50: strength, mobility and low-impact classes at off-peak times in big-city clubs | 403 Established members (41%) are 50 or older and 54 are 60 or older; Starters include just one member aged 50 or over | Class fill rate and over-50 retention |
| Data & sign-up | Drop income from the sign-up form, and let the plan a member chooses signal what they're willing to pay | Income follows age almost exactly (r = 0.9995), and it's one of the most personal questions a member is asked | Sign-up completion rate |
| Data & sign-up | Assign segments from city and job type: big city, or a skilled or management job, means Established | The rule matches the clustering for 97.8% of members; dropping gender and marital status keeps 99.4% in the same segment | Rule against a quarterly re-cluster |
| Data & sign-up | Start recording behaviour: visits per week, class bookings, plan, tenure and the reason for every cancellation | All seven fields describe one life-stage axis; none says who visits, who leaves or what they'd pay for | Members with 90 days of visit data |

## Long-term strategy: 1–3 years

| Track | Now (0–6 months) | Next (6–18 months) | Later (18–36 months) |
|---|---|---|---|
| **Know the member** (marketing, data & CRM) | Behaviour on file: visits, class bookings, plan and tenure for every member, and a reason for every cancellation | Segments that predict: re-segment within each life stage on visits, class mix and tenure, and keep only segments that differ in 90-day retention | Offers sized to value: predict churn and lifetime value, size retention offers to each member's value, and test them against a held-out control group |
| **Shape the clubs** (club operations, product) | Formats by city: the Starter format in small and mid-sized cities, the Established format in big cities, and a premium option in small-city clubs | Test the plan menu: pilot the three tiers in matched pairs of clubs, and keep the version that lifts joins and 90-day retention | Plan the network: use each city's segment mix and growth to decide where to refit, open or add express formats near workplaces |

**Member scorecard:** 90-day retention by segment · visits per member per week · upgrades between tiers.
**Club scorecard:** joins by club format · class fill rate, including over-50 classes · members joining through employer agreements.

---

## What carries over

- **Read the fields before the clusters.** Income repeated age (r = 0.9995). Spotting that first explains why every method agreed.
- **Report proportions, not modes.** A cluster labelled "male" was 36.5% female. Percentages keep segment names honest.
- **Agreement isn't insight.** Methods agree when the data has one axis. Whether a segment matters depends on behaviour this data doesn't have.
- **Ask for less.** The segments held for 99.4% of members without gender or marital status.

**What I'd change next time:** use a distance built for mixed data, such as Gower or k-prototypes, from the start; judge segments by an outcome, such as retention or visits, not by silhouette alone; collect behaviour first, then test offers against a control group before rolling them out; re-run the segmentation on each quarter's new members to see whether it holds.

## Repository

| File | What it is |
|---|---|
| `Task.ipynb` | The original analysis: data checks, EDA, elbow and silhouette plots, k-means++ and Ward clustering, and the overlap between them |
| `deep_dive.py` | The deeper analysis behind the case study: proportions, stress tests, single-field rules, the city split and the members between the segments; writes every figure on the page to one JSON file |
| `Report.pdf` | The original report for management |
| `data/data.csv` | 2,000 members, seven fields |
| `data/data legend.csv` | What each field and code means |
| `gym-segmentation.jpg` | Cover image: every member, stacked by age and coloured by segment |
| `index.html` | Case study page, served at [racheltran184.github.io/Python-Project-3](https://racheltran184.github.io/Python-Project-3/) |
| `LICENSE` | Apache License 2.0 |

## How to run

1. Install Python 3.11 or later with `pandas`, `numpy`, `scikit-learn`, `matplotlib` and `seaborn`.
2. Run **`Task.ipynb`** from the repository root. It reads `data/` and saves the report's figures as PNG files next to the notebook.
3. Run **`python deep_dive.py data site_data.json`** to rebuild every figure on the case study page. It takes a few seconds.

## Data and privacy

The 2,000 member records came with the brief: a teaching case, not a real gym chain's data. Income follows age almost exactly (r = 0.9995), which suggests generated data, so the figures describe this dataset and the recommendations are illustrative. An individual student project for a postgraduate predictive analytics unit.

---

**Rachel Tran** · [Portfolio](https://racheltran.vercel.app) · [LinkedIn](https://www.linkedin.com/in/racheltran184/) · [GitHub](https://github.com/racheltran184)
