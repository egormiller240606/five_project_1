## Critique 1: the mechanisms are claimed, not identified

The main economic conclusions of the paper do not really provide anything substantial. The abstract states that the dominant presence of retail investors positively affects short-term predictability, and the conclusion states that government signaling drives the long-horizon predictability of SOEs. However, these claims are tested only by splitting the sample into subgroups, and all three splits (size, A.M.C.P.S., SOE) reduce to firm size.

- SOE vs non-SOE: the authors themselves note in Section 3.1.4 that company size is strongly correlated with SOE status, since SOEs are the dominant firms in banking, infrastructure and military.
- Retail vs institutional: the proxy A.M.C.P.S. = Market Cap / Number of Shareholders has market capitalization in the numerator. Two firms with the same 50,000 shareholders but a market cap of 1 bn and 100 bn fall into different groups only because of their size. The "retail" split therefore mechanically repeats the size split.
- Government signaling is never measured. None of the 1,160 predictors captures state support, political connections or policy exposure. The explanation is added after observing that SOEs are more predictable at the annual horizon.

If these facts are not taken into account, the main conclusion is simply that "machine learning predicts returns of small Chinese stocks well", which is pretty narrow and does not completely cover such complicated topic.

How to check: double sorts on the authors' own data. Within each size quintile, compare predictability of SOEs and non-SOEs, and of high and low A.M.C.P.S. stocks. If the differences disappear, the "state" and "retail" channels are a size effect.

## Critique 2: the results describe one regulatory regime, which no longer exists

The authors explain predictability through institutional features of the Chinese market: approval-based IPOs (which create shell value in the bottom 30% of stocks), short-sale restrictions, retail dominance and daily price limits. These are policy decisions, not permanent properties of the market. A model trained on 2000–2020 learns this regime, while the conclusion generalizes that machine learning can be (even more) successfully applied to markets that differ from the US.

- 42% of the long-only NN4 return comes from the bottom 30% of stocks (4.50% per month in Table 6 vs 2.60% in Table 7), and the authors link these stocks to shell value caused by IPO restrictions (Section 4.2).
- The regime was already changing inside the sample: the authors report a structural break in variable importance around 2015 and a drop in R² in 2018 due to the trade war.
- In February 2023 China fully switched to a registration-based IPO system, removing the approval regime that created shell value.
- The same small-cap exposure became a crowded quant trade and collapsed in February 2024: in the first week of February the CSI 300 rose by 5.8%, while a popular microcap index fell by 17.5% as state-backed investors supported large caps. Regulators then restricted short selling and tightened rules on quant trading.

In a market where the regulator shapes the return-generating process, predicting returns is partly predicting policy. The paper does not separate structural predictability from predictability that depends on the rules for IPOs, short selling and price limits.

How to check: replicate a simple long-only liquidity sort on a new period (2021–2025) with free data (e.g., akshare) and examine its performance after the 2023 IPO reform and in February 2024.