"""Feature catalog from the paper's Internet Appendix (Tables C.1/C.4/C.5)."""

STOCK_CHARACTERISTICS = [
    "absacc","acc","agr","beta","betasq","bm","bm_ia","cash","cashdebt","cashspr",
    "cfp","cfp_ia","chato","chatoia","chcsho","chempia","chinv","chmom","chpm","chpmia",
    "chtx","cinvest","currat","depr","divi","divo","dolvol","dy","ear","egr","gma","grCAPX",
    "herf","hire","idiovol","ill","invest","lev","lgr","maxret","mom12m","mom1m","mom6m",
    "mom36m","ms","mve","mve_ia","nincr","operprof","orgcap","pchcapx_ia","pchcurrat",
    "pchdepr","pchgm_pchsale","pchquick","pchsale_pchinvt","pchsale_pchrect","pchsale_pchxsga",
    "pchsaleinv","pctacc","pricedelay","ps","quick","rd","rd_mve","rd_sale","realestate",
    "volatility","roaq","roavol","roeq","roic","rsup","salecash","saleinv","salerev","sgr",
    "sp","std_dolvol","std_turn","stdacc","stdcf","tang","tb","turn","zerotrade","atr",
    "er_trend","largestholderrate","top10holderrate"
]

OWNERSHIP_DUMMIES = ["soe", "private", "foreign", "others"]
MACRO_VARIABLES = ["dp", "de", "bm", "svar", "ep", "ntis", "tms", "infl", "mtr", "m2gr", "itgr"]

# Paper groups (minor typography normalized to valid Python column names).
CHARACTERISTIC_GROUPS = {
    "size": ["mve", "mve_ia", "herf", "chinv", "chcsho"],
    "beta": ["beta", "betasq"],
    "momentum": ["mom1m", "mom6m", "mom12m", "mom36m", "chmom", "er_trend", "maxret"],
    "liquidity": ["std_dolvol", "zerotrade", "atr", "chatoia", "std_turn", "ill", "turn", "dolvol", "pricedelay"],
    "volatility": ["idiovol", "ear", "volatility", "roavol"],
    "ownership": ["top10holderrate", "largestholderrate"],
    "book_to_price": ["bm", "bm_ia", "cfp", "cfp_ia", "sp", "cashspr", "invest", "realestate", "depr"],
    "earnings": ["roeq", "roaq", "divo", "absacc", "divi", "salerev", "chempia", "nincr", "chpmia", "stdacc", "chtx", "cash", "roic", "chpm", "stdcf", "chato", "dy", "acc", "pctacc", "saleinv", "operprof", "pchsale_pchrect", "salecash", "tb", "gma", "pchdepr"],
    "growth": ["egr", "orgcap", "sgr", "pchgm_pchsale", "rsup", "pchsaleinv", "rd_sale", "rd_mve", "rd", "cinvest", "pchsale_pchxsga", "pchsale_pchinvt", "agr", "grCAPX", "hire"],
    "leverage": ["lev", "pchquick", "pchcapx_ia", "lgr", "quick", "ps", "tang", "currat", "ms", "pchcurrat", "cashdebt"],
}

# Frequency counts reported by the paper: 22 monthly, 51 quarterly, 6 semi-annual, 15 annual (includes ownership flags).
FREQUENCY = {
    **{x:"monthly" for x in ["beta","betasq","chcsho","chmom","dolvol","idiovol","ill","maxret","mom12m","mom1m","mom6m","mom36m","mve","mve_ia","pricedelay","volatility","std_dolvol","std_turn","turn","zerotrade","atr","er_trend"]},
    **{x:"semiannual" for x in ["absacc","acc","depr","grCAPX","pchdepr","pctacc"]},
    **{x:"annual" for x in ["agr","chempia","divi","divo","dy","hire","invest","ms","pchcapx_ia","largestholderrate","top10holderrate","soe","private","foreign","others"]},
}
for _x in STOCK_CHARACTERISTICS + OWNERSHIP_DUMMIES:
    FREQUENCY.setdefault(_x, "quarterly")
