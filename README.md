# Machine-Learning in the Chinese Factor Zoo

This repo has the paper I read for my project: **"Machine Learning in the Chinese Factor Zoo"** by Leippold, Wang & Zhou (2022, *Journal of Financial Economics*). Basically it's an extension of the famous Gu, Kelly & Xiu (2020) "empirical asset pricing via machine learning" paper, but applied to the Chinese stock market instead of the US.

## What they did

The authors take 13 different models: from plain OLS regression, to LASSO/Elastic Net/PLS, to tree-based methods (Random Forest, Gradient Boosted Trees), all the way up to neural networks with 1 to 5 hidden layers. They use them to predict Chinese stock returns. They built a huge set of predictors (1,160 signals in total) using 94 stock characteristics, 11 macro variables, and industry dummies, plus a few factors made specifically for the Chinese market to capture retail speculative trading. Data covers 2000–2020.

## Main takeaways

- **Neural networks win** — they give the best and most stable out-of-sample R² across pretty much every test the authors run.
- **China shows way more predictability than the US** — the R² numbers here are several times higher than in the original US study.
- **Liquidity matters more than momentum**, which is the opposite of what's usually found in the US market.
- **Small stocks / retail-driven names are easier to predict short-term**, while **big stocks and state-owned enterprises (SOEs) are easier to predict long-term** — this ties back to retail investors dominating short-term trading, and the government influencing SOEs over longer horizons.
- Even after accounting for transaction costs, price limits, and the fact that short-selling is basically impossible in China, **long-only portfolios built from ML predictions still perform really well**.

Overall a pretty interesting read since it shows machine learning isn't just a "US market thing" — it actually works even better in a market with different structure and more retail-driven noise.
