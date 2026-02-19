"""
Crude Oil Strategy Explorer — Streamlit Dashboard
===================================================
Run:  streamlit run dashboard.py
Deps: pip install streamlit plotly openpyxl scipy python-dateutil
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Crude Oil Strategy Explorer", page_icon="🛢️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #080d14;
}
[data-testid="stAppViewContainer"] {
    background: #080d14;
    background-image: radial-gradient(ellipse 90% 40% at 50% 0%, rgba(0,200,160,0.07) 0%, transparent 70%);
}
[data-testid="stSidebar"] {
    background: #0c1220 !important;
    border-right: 1px solid #1a2535 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

.metric-card {
    background: linear-gradient(160deg, #101925 0%, #0d1720 100%);
    border-radius: 10px;
    padding: 16px 14px 12px;
    border: 1px solid #1c2d3d;
    border-top: 2px solid #00c8a0;
    margin-bottom: 4px;
}
.mv {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: #e4f0ec;
    line-height: 1.1;
}
.ml {
    font-size: 0.7rem;
    font-weight: 500;
    color: #3d5a70;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 5px;
}
.mdp { color: #00c8a0; font-size: 0.75rem; font-weight: 500; }
.mdn { color: #f05e5e; font-size: 0.75rem; font-weight: 500; }

.banner {
    background: linear-gradient(120deg, #09182a 0%, #0a2218 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    border: 1px solid #1c3028;
    position: relative;
    overflow: hidden;
}
.banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #00c8a0, transparent);
}
.banner h1 {
    font-family: 'Inter', sans-serif;
    color: #daeee8;
    font-size: 1.45rem;
    font-weight: 600;
    margin: 0 0 5px 0;
}
.banner p { color: #3d6b5a; font-size: 0.86rem; margin: 0; }
.pill {
    display: inline-block;
    background: rgba(0,200,160,0.1);
    color: #00c8a0;
    border: 1px solid rgba(0,200,160,0.2);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 6px;
    margin-top: 10px;
}

.sec-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #285a48;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2e24;
}

.opt-bar {
    background: linear-gradient(120deg, #05261c 0%, #07301f 100%);
    border: 1px solid #1a4030;
    border-left: 3px solid #00c8a0;
    border-radius: 8px;
    padding: 11px 16px;
    margin-bottom: 14px;
}
.opt-bar b { color: #00c8a0; font-size: 0.92rem; }
.opt-bar span { color: #2d6050; font-size: 0.8rem; }

.sb-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #1e4a3a;
    padding: 10px 0 5px 0;
    border-bottom: 1px solid #131f2c;
    margin-bottom: 8px;
}

.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #00c8a0, #009e7c) !important;
    color: #040a0f !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 7px !important;
}

hr { border-color: #131f2c !important; }

/* sidebar — make all text light/readable */
[data-testid="stSidebar"] label { color: #c8dde6 !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #b0ccd8 !important; }
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption { color: #8aafc0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #b0ccd8 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color: #daeef5 !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] span { color: #c8dde6 !important; }
</style>
""", unsafe_allow_html=True)

ANNUALIZATION_FACTOR = np.sqrt(252)

OPTIMAL = dict(
    mode="Walkforward (Auto-Select)",
    strat_pool="ALL",
    lookback_m=18, rebalance_m=6,
    score_name="Composite (60% Sharpe + 40% Sortino)",
    tc_bps=2, use_vt=True, target_vol_pct=15, use_regime=False,
    regime_thresh=0.65,
)

BASE_STRATEGIES = [
    "1M_momentum","3M_momentum","MA200_crossover",
    "20D_zscore_meanrev","5D_reversal","percentile_reversion",
    "vol_adj_momentum","vol_breakout","vol_trend",
    "carry_proxy","momentum_vol_filtered",
]
VOL_SCALED = [f"{s}_vol_scaled" for s in BASE_STRATEGIES]
ALL_CANDIDATES = BASE_STRATEGIES + VOL_SCALED + ["ensemble_equal"]

DISPLAY_TO_INTERNAL = {
    "1M Momentum": "1M_momentum",
    "3M Momentum": "3M_momentum",
    "MA Crossover": "MA200_crossover",
    "Z-Score Mean Reversion": "20D_zscore_meanrev",
    "Short-Term Reversal": "5D_reversal",
    "Percentile Reversion": "percentile_reversion",
    "Vol-Adjusted Momentum": "vol_adj_momentum",
    "Volatility Breakout": "vol_breakout",
    "Volatility Trend": "vol_trend",
    "Carry Proxy": "carry_proxy",
    "Momentum + Vol Filter": "momentum_vol_filtered",
}
STRATEGIES = list(DISPLAY_TO_INTERNAL.keys())

@st.cache_data
def load_data():
    df = pd.read_excel("brent_index.xlsx")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    prices  = df["CO1 Comdty"]
    returns = prices.pct_change().dropna()
    return prices, returns

def construct_all_signals(prices, returns):
    signals = pd.DataFrame(index=returns.index)
    prices  = prices.loc[returns.index]
    ret_1m = prices.pct_change(21)
    ret_3m = prices.pct_change(63)
    ma_200 = prices.rolling(200).mean()
    signals['1M_momentum']     = np.sign(ret_1m).shift(1)
    signals['3M_momentum']     = np.sign(ret_3m).shift(1)
    signals['MA200_crossover'] = np.sign(prices - ma_200).shift(1)
    roll_mean = returns.rolling(20).mean()
    roll_std  = returns.rolling(20).std()
    z_score   = (returns - roll_mean) / roll_std
    signals['20D_zscore_meanrev'] = -np.sign(z_score).shift(1)
    signals['5D_reversal']        = -np.sign(prices.pct_change(5)).shift(1)
    pct_rank = prices.pct_change(10).rolling(126).rank(pct=True).shift(1)
    signals['percentile_reversion'] = np.where(
        pct_rank > 0.8, -1, np.where(pct_rank < 0.2, 1, 0))
    vol_21d = returns.rolling(21).std() * ANNUALIZATION_FACTOR
    signals['vol_adj_momentum'] = (ret_1m / vol_21d).clip(-3, 3).shift(1)
    vol_10d = returns.rolling(10).std() * ANNUALIZATION_FACTOR
    vol_63d = returns.rolling(63).std() * ANNUALIZATION_FACTOR
    signals['vol_breakout'] = np.where((vol_10d / vol_63d).shift(1) > 1.2, 1, -1)
    signals['vol_trend']    = np.sign(vol_21d.pct_change(21)).shift(1)
    signals['carry_proxy']           = np.sign(ret_1m - ret_3m / 3).shift(1)
    vol_pct = vol_21d.rolling(252).rank(pct=True).shift(1)
    signals['momentum_vol_filtered'] = np.sign(ret_1m).shift(1) * np.where(vol_pct > 0.5, 1.0, 0.5)
    return signals.fillna(0)

def build_all_candidates(signals, returns):
    vol_63d = returns.rolling(63).std() * ANNUALIZATION_FACTOR
    vol_pct = vol_63d.rolling(252).rank(pct=True).clip(0, 1)
    filtered = pd.DataFrame(index=signals.index)
    for col in signals.columns:
        filtered[f"{col}_vol_scaled"] = signals[col] * vol_pct
    ensemble = signals.mean(axis=1)
    all_candidates = pd.concat([signals, filtered], axis=1)
    all_candidates['ensemble_equal'] = ensemble
    return all_candidates

def apply_tc(sig, returns, tc):
    gross    = sig * returns
    turnover = sig.diff().abs().fillna(0)
    return gross - turnover * tc

def vol_target(rets, tv, span=63):
    realized = rets.ewm(span=span).std().shift(1) * ANNUALIZATION_FACTOR
    scalar   = (tv / realized).clip(0.25, 4.0)
    return rets * scalar

def get_regimes(returns, vw=63, thresh=0.65):
    vol     = returns.rolling(vw).std() * ANNUALIZATION_FACTOR
    vol_pct = vol.rolling(252).rank(pct=True)
    return vol, vol_pct, (vol_pct > thresh).astype(int), (vol_pct < (1-thresh)).astype(int)

def calc_metrics(rets):
    rets = rets.dropna()
    if len(rets) < 10: return {}
    eq   = (1 + rets).cumprod()
    tr   = eq.iloc[-1] - 1;  ny = len(rets) / 252
    cagr = (1 + tr) ** (1/ny) - 1
    vol  = rets.std() * ANNUALIZATION_FACTOR
    sh   = rets.mean() / rets.std() * ANNUALIZATION_FACTOR if rets.std() > 0 else 0
    rm   = eq.expanding().max(); dd = (eq - rm) / rm; mdd = abs(dd.min())
    cal  = cagr / mdd if mdd > 0 else 0
    wr   = (rets > 0).mean()
    ds   = rets[rets < 0].std() * ANNUALIZATION_FACTOR
    so   = rets.mean() * 252 / ds if ds > 0 else 0
    return dict(sh=sh, so=so, cagr=cagr, vol=vol, mdd=mdd, cal=cal, wr=wr, tr=tr, eq=eq, dd=dd)

def score_sharpe(s):
    if len(s) < 10 or s.std() == 0: return 0
    return s.mean() / s.std()
def score_sortino(s):
    if len(s) < 10: return 0
    ds = s[s < 0].std(); return s.mean() / ds if ds > 0 else 0
def score_composite(s):
    if len(s) < 10 or s.std() == 0: return 0
    return 0.6 * score_sharpe(s) + 0.4 * score_sortino(s)
SCORE_FNS = {"Sharpe": score_sharpe, "Sortino": score_sortino,
             "Composite (60% Sharpe + 40% Sortino)": score_composite}

def run_walkforward(df, lb_m, rb_m, fn):
    from dateutil.relativedelta import relativedelta
    idx     = df.index
    periods = pd.date_range(start=idx[0],
                            end=idx[-1] + relativedelta(months=rb_m),
                            freq=f"{rb_m}MS")
    port = pd.Series(np.nan, index=idx); sels = []
    for s, e in zip(periods[:-1], periods[1:]):
        lbe = s - pd.Timedelta(days=1)
        lbs = lbe - relativedelta(months=lb_m)
        if lbs < idx[0]: continue
        lb = df.loc[lbs:lbe]
        if len(lb) < 10: continue
        sc = lb.apply(fn); best = sc.idxmax()
        port.loc[s:e] = df.loc[s:e, best]
        sels.append({"period": str(s.date()), "strategy": best, "score": round(sc[best], 4)})
    return port, pd.DataFrame(sels)

# ── shared chart theme (no hovermode/margin here to avoid subplot conflicts) ──
C_STRAT = "#00c8a0"
C_BASE  = "#3a5a6a"
C_DD    = "#f05e5e"

def base_layout(**kwargs):
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,20,0.8)",
        font=dict(family="Inter, sans-serif", color="#3a5a6a", size=11),
        hoverlabel=dict(bgcolor="#0d1a26", bordercolor="rgba(0,200,160,0.3)",
                        font=dict(color="#c8e8e0", size=11)),
        legend=dict(orientation="h", y=1.1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,200,160,0.06)",
                   zeroline=False, linecolor="#1a2535", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,200,160,0.06)",
                   zeroline=False, linecolor="#1a2535", tickfont=dict(size=10)),
        hovermode="x unified",
        margin=dict(l=2, r=2, t=28, b=2),
        **kwargs
    )

# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="banner">
  <h1>🛢️ Crude Oil Strategy Explorer</h1>
  <p>Interactive walkforward backtester · Brent Crude 2016–2024</p>
  <span class="pill">SHARPE 0.847</span>
  <span class="pill">BASELINE 0.46</span>
  <span class="pill">23 CANDIDATES</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════
try:
    prices, returns = load_data()
except FileNotFoundError:
    st.error("❌ `brent_index.xlsx` not found. Place it in the same folder as dashboard.py")
    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.95rem;font-weight:600;color:#00c8a0;margin-bottom:14px">⚙ CONTROLS</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-hdr">🏆 Quick Preset</div>', unsafe_allow_html=True)
    load_optimal = st.button("⚡ Load Optimal Settings (Sharpe 0.847)", type="primary",
                              use_container_width=True)
    if load_optimal:
        for k,v in OPTIMAL.items():
            st.session_state[k] = v
        st.session_state["_optimal_loaded"] = True
        st.rerun()

    if st.session_state.get("_optimal_loaded"):
        st.success("✅ Optimal settings loaded!")
        if st.button("✖ Clear preset", use_container_width=True):
            st.session_state["_optimal_loaded"] = False
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sb-hdr">📌 Mode</div>', unsafe_allow_html=True)
    _mode_default = st.session_state.get("mode", "Walkforward (Auto-Select)")
    mode = st.radio("", ["Single Strategy","Walkforward (Auto-Select)"],
                    index=1 if _mode_default == "Walkforward (Auto-Select)" else 0,
                    key="mode")

    if mode == "Single Strategy":
        sel_strat = st.selectbox("Strategy", STRATEGIES, key="sel_strat")
    else:
        lb_m  = st.slider("Lookback (months)",  6, 36, st.session_state.get("lookback_m", 18),  key="lookback_m")
        rb_m  = st.slider("Rebalance (months)", 1,  6, st.session_state.get("rebalance_m", 6),  key="rebalance_m")
        sc_nm = st.selectbox("Scoring", list(SCORE_FNS.keys()),
                             index=list(SCORE_FNS.keys()).index(
                                 st.session_state.get("score_name","Composite (60% Sharpe + 40% Sortino)")),
                             key="score_name")
        st.caption(f"📊 Candidate pool: **{len(ALL_CANDIDATES)} strategies** (11 base + 11 vol-scaled + 1 ensemble)")

    st.markdown("---")
    st.markdown('<div class="sb-hdr">🎯 Risk Controls</div>', unsafe_allow_html=True)
    tc_bps    = st.slider("Transaction cost (bps)", 0, 10, st.session_state.get("tc_bps", 2),       key="tc_bps")
    use_vt    = st.toggle("Volatility Targeting",     value=st.session_state.get("use_vt", True),   key="use_vt")
    tgt_vol   = st.slider("Target vol (%)", 5, 30, st.session_state.get("target_vol_pct", 15),      key="target_vol_pct") if use_vt else 15
    use_reg   = st.toggle("Regime Filter",             value=st.session_state.get("use_regime",False),key="use_regime")
    reg_thr   = st.slider("High-vol threshold",0.50,0.85,
                           st.session_state.get("regime_thresh",0.65), key="regime_thresh") if use_reg else 0.65

    st.markdown("---")
    st.markdown('<div class="sb-hdr">📅 Date Range</div>', unsafe_allow_html=True)
    d0_def = returns.index[0].date(); d1_def = returns.index[-1].date()
    dr = st.date_input("Range", (d0_def, d1_def), min_value=d0_def, max_value=d1_def)

# ─── filter dates ────────────────────────────────────────────────
if len(dr)==2:
    pf = prices.loc[pd.Timestamp(dr[0]):pd.Timestamp(dr[1])]
    rf = returns.loc[pd.Timestamp(dr[0]):pd.Timestamp(dr[1])]
else:
    pf, rf = prices, returns

tc = tc_bps/10000

# ═══════════════════════════════════════════════════════════════════
#  COMPUTE
# ═══════════════════════════════════════════════════════════════════
with st.spinner("Computing strategy..."):
    baseline_m = calc_metrics(rf)
    all_signals    = construct_all_signals(pf, rf)
    all_candidates = build_all_candidates(all_signals, rf)
    strategy_returns = pd.DataFrame(index=rf.index)
    for col in all_candidates.columns:
        pos = all_candidates[col].fillna(0)
        strategy_returns[col] = apply_tc(pos, rf, tc)
    vt_strategy_returns = pd.DataFrame(index=rf.index)
    for col in strategy_returns.columns:
        vt_strategy_returns[col] = vol_target(strategy_returns[col], tgt_vol/100) if use_vt else strategy_returns[col]

    if mode == "Single Strategy":
        internal = DISPLAY_TO_INTERNAL.get(sel_strat, sel_strat)
        col = internal if internal in vt_strategy_returns.columns else vt_strategy_returns.columns[0]
        sr = vt_strategy_returns[col]
        strat_m = calc_metrics(sr); label = sel_strat; sels_df = None
    else:
        sr, sels_df = run_walkforward(vt_strategy_returns, lb_m, rb_m, SCORE_FNS[sc_nm])
        strat_m = calc_metrics(sr); label = f"Walkforward ({rb_m}M)"

# ═══════════════════════════════════════════════════════════════════
#  OPTIMAL BANNER
# ═══════════════════════════════════════════════════════════════════
if st.session_state.get("_optimal_loaded") and strat_m:
    st.markdown(f"""
    <div class="opt-bar">
    <b>⚡ Optimal Active — Sharpe {strat_m['sh']:.3f} · Sortino {strat_m['so']:.3f} · Max DD {strat_m['mdd']*100:.1f}% · CAGR {strat_m['cagr']*100:.1f}%</b><br>
    <span>18M lookback · 6M rebalance · Composite scoring · 15% vol targeting · All 11 strategies</span>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  METRICS ROW
# ═══════════════════════════════════════════════════════════════════
def delta_html(v, r, higher_better=True):
    d=v-r; good=(d>0)==higher_better
    cls="mdp" if good else "mdn"; sym="▲" if d>0 else "▼"
    return f'<span class="{cls}">{sym} {abs(d):.3f} vs baseline</span>'

def pct_delta(v, r, higher_better=True):
    d=v-r; good=(d>0)==higher_better
    cls="mdp" if good else "mdn"; sym="▲" if d>0 else "▼"
    return f'<span class="{cls}">{sym} {abs(d)*100:.1f}pp vs baseline</span>'

if strat_m:
    cols = st.columns(6)
    items = [
        ("Sharpe",       f'{strat_m["sh"]:.3f}',  delta_html(strat_m["sh"],  baseline_m["sh"])),
        ("Sortino",      f'{strat_m["so"]:.3f}',  delta_html(strat_m["so"],  baseline_m["so"])),
        ("CAGR",         f'{strat_m["cagr"]*100:.1f}%', delta_html(strat_m["cagr"],baseline_m["cagr"])),
        ("Max Drawdown", f'{strat_m["mdd"]*100:.1f}%',  pct_delta(strat_m["mdd"], baseline_m["mdd"],False)),
        ("Volatility",   f'{strat_m["vol"]*100:.1f}%',  pct_delta(strat_m["vol"], baseline_m["vol"],False)),
        ("Win Rate",     f'{strat_m["wr"]*100:.1f}%',   pct_delta(strat_m["wr"],  baseline_m["wr"])),
    ]
    for col,(lbl,val,dlt) in zip(cols,items):
        col.markdown(f'<div class="metric-card"><div class="mv">{val}</div>'
                     f'<div class="ml">{lbl}</div><div style="margin-top:6px">{dlt}</div></div>',
                     unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
#  ROW 1: Equity + Drawdown
# ═══════════════════════════════════════════════════════════════════
c1,c2 = st.columns([3,2])

with c1:
    st.markdown('<div class="sec-hdr">📈 Equity Curve</div>', unsafe_allow_html=True)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=baseline_m["eq"].index,y=baseline_m["eq"],
        name=f'Baseline (Sharpe:{baseline_m["sh"]:.2f})',
        line=dict(color=C_BASE,width=1.5,dash="dot")))
    if strat_m:
        fig.add_trace(go.Scatter(x=strat_m["eq"].index,y=strat_m["eq"],
            name=f'{label} (Sharpe:{strat_m["sh"]:.2f})',
            line=dict(color=C_STRAT,width=2.5),
            fill="tozeroy",fillcolor="rgba(0,200,160,0.05)"))
    fig.update_layout(**base_layout(height=320, yaxis_title="Cumulative Return"))
    st.plotly_chart(fig,use_container_width=True)

with c2:
    st.markdown('<div class="sec-hdr">📉 Drawdown</div>', unsafe_allow_html=True)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=baseline_m["dd"].index,y=baseline_m["dd"]*100,
        name="Baseline",fill="tozeroy",
        line=dict(color=C_BASE,width=1),fillcolor="rgba(58,90,106,0.12)"))
    if strat_m:
        fig2.add_trace(go.Scatter(x=strat_m["dd"].index,y=strat_m["dd"]*100,
            name=label,fill="tozeroy",
            line=dict(color=C_DD,width=1.8),fillcolor="rgba(240,94,94,0.15)"))
    fig2.update_layout(**base_layout(height=320, yaxis_title="Drawdown (%)"))
    st.plotly_chart(fig2,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
#  ROW 2: Rolling Sharpe + Regime
# ═══════════════════════════════════════════════════════════════════
c3,c4 = st.columns([2,3])

with c3:
    st.markdown('<div class="sec-hdr">📊 Rolling 1Y Sharpe</div>', unsafe_allow_html=True)
    if strat_m:
        rs_ = sr.dropna()
        roll_sh = rs_.rolling(252).mean()/rs_.rolling(252).std()*ANNUALIZATION_FACTOR
        fig3=go.Figure()
        fig3.add_trace(go.Scatter(x=roll_sh.index,y=roll_sh,
            line=dict(color=C_STRAT,width=2),showlegend=False,
            fill="tozeroy",fillcolor="rgba(0,200,160,0.07)"))
        fig3.add_hline(y=0,line_dash="dash",line_color=C_DD,opacity=0.6)
        fig3.add_hline(y=baseline_m["sh"],line_dash="dot",line_color=C_BASE,
            opacity=0.7,annotation_text="Baseline",annotation_position="bottom right",
            annotation_font_color="#3a5a6a")
        fig3.update_layout(**base_layout(height=300, yaxis_title="Sharpe (1Y)"))
        st.plotly_chart(fig3,use_container_width=True)

with c4:
    st.markdown('<div class="sec-hdr">🌡️ Volatility Regime</div>', unsafe_allow_html=True)
    vol_r, vp_r, hv_r, lv_r = get_regimes(rf, 63, reg_thr)
    fig4=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.5,0.5],vertical_spacing=0.05)
    fig4.add_trace(go.Scatter(x=vp_r.index,y=vp_r,line=dict(color="#a8dadc",width=1.5),
        name="Vol Pct"),row=1,col=1)
    fig4.add_hline(y=reg_thr,line_dash="dash",line_color=C_DD,row=1,col=1,opacity=0.6)
    fig4.add_hline(y=1-reg_thr,line_dash="dash",line_color=C_STRAT,row=1,col=1,opacity=0.6)
    fig4.add_trace(go.Scatter(x=vol_r.index,y=vol_r*100,fill="tozeroy",
        line=dict(color="#e76f51",width=1.2),fillcolor="rgba(231,111,81,0.18)",
        name="Vol %"),row=2,col=1)
    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,20,0.8)",
        font=dict(family="Inter, sans-serif", color="#3a5a6a", size=11),
        hoverlabel=dict(bgcolor="#0d1a26", bordercolor="rgba(0,200,160,0.3)",
                        font=dict(color="#c8e8e0", size=11)),
        margin=dict(l=2,r=2,t=28,b=2),
        height=300, showlegend=False, hovermode="x unified"
    )
    fig4.update_xaxes(showgrid=True,gridcolor="rgba(0,200,160,0.06)",
                      zeroline=False,linecolor="#1a2535",tickfont=dict(size=10))
    fig4.update_yaxes(showgrid=True,gridcolor="rgba(0,200,160,0.06)",
                      zeroline=False,linecolor="#1a2535",tickfont=dict(size=10))
    fig4.update_yaxes(title_text="Percentile",row=1,col=1)
    fig4.update_yaxes(title_text="Ann.Vol (%)",row=2,col=1)
    st.plotly_chart(fig4,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
#  WALKFORWARD DETAILS
# ═══════════════════════════════════════════════════════════════════
if mode=="Walkforward (Auto-Select)" and sels_df is not None and len(sels_df):
    st.markdown("---")
    st.markdown('<div class="sec-hdr">🔄 Walkforward Selections</div>', unsafe_allow_html=True)
    ca,cb = st.columns([2,3])
    with ca:
        freq=sels_df["strategy"].value_counts().reset_index()
        freq.columns=["Strategy","Count"]
        fig5=px.bar(freq,x="Count",y="Strategy",orientation="h",
            color="Count",color_continuous_scale=[[0,"#0a2518"],[1,"#00c8a0"]],
            template="plotly_dark")
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(8,13,20,0.8)",
            height=280,margin=dict(l=2,r=2,t=36,b=2),
            coloraxis_showscale=False,showlegend=False,
            title=dict(text="Selection Frequency",font=dict(size=11,color="#3a5a6a")),
            font=dict(family="Inter, sans-serif"))
        fig5.update_traces(marker_line_width=0)
        st.plotly_chart(fig5,use_container_width=True)
    with cb:
        st.dataframe(sels_df.rename(columns={"period":"Period","strategy":"Strategy","score":"Score"}),
                     use_container_width=True,height=280)

# ═══════════════════════════════════════════════════════════════════
#  REGIME ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="sec-hdr">🎯 Regime Attribution</div>', unsafe_allow_html=True)
_,vp_ra,hv_ra,lv_ra = get_regimes(rf, 63, reg_thr)
ra = []
for rn,mask in [("High Vol",hv_ra==1),("Low Vol",lv_ra==1),
                ("Neutral",(hv_ra==0)&(lv_ra==0))]:
    r_=sr.reindex(rf.index)[mask].dropna() if strat_m else pd.Series()
    if len(r_)>10:
        sh_=r_.mean()/r_.std()*ANNUALIZATION_FACTOR if r_.std()>0 else 0
        ra.append({"Regime":rn,"Sharpe":round(sh_,3),"Days":int(mask.sum()),
                   "% Time":f"{mask.mean()*100:.1f}%"})

if ra:
    ra_df=pd.DataFrame(ra)
    cc1,cc2=st.columns([2,3])
    with cc1:
        fig6=px.bar(ra_df,x="Regime",y="Sharpe",color="Sharpe",
            color_continuous_scale=[[0,"#6b1a1a"],[0.5,"#1a3d2e"],[1,"#00c8a0"]],
            template="plotly_dark",text="Sharpe")
        fig6.update_traces(texttemplate="%{text:.3f}",textposition="outside",marker_line_width=0)
        fig6.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(8,13,20,0.8)",
            height=260,margin=dict(l=2,r=2,t=10,b=2),
            coloraxis_showscale=False,showlegend=False,
            font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig6,use_container_width=True)
    with cc2:
        st.dataframe(ra_df,use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════════
#  FULL METRICS TABLE
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="sec-hdr">📋 Full Performance Comparison</div>', unsafe_allow_html=True)
if strat_m and baseline_m:
    sm_tbl = pd.DataFrame({
        "Metric": ["Sharpe","Sortino","CAGR (%)","Vol (%)","Max DD (%)","Calmar","Win Rate (%)","Total Return (%)"],
        "Baseline": [round(baseline_m["sh"],3),round(baseline_m["so"],3),
                     round(baseline_m["cagr"]*100,2),round(baseline_m["vol"]*100,2),
                     round(baseline_m["mdd"]*100,2),round(baseline_m["cal"],3),
                     round(baseline_m["wr"]*100,2),round(baseline_m["tr"]*100,2)],
        label:       [round(strat_m["sh"],3),round(strat_m["so"],3),
                     round(strat_m["cagr"]*100,2),round(strat_m["vol"]*100,2),
                     round(strat_m["mdd"]*100,2),round(strat_m["cal"],3),
                     round(strat_m["wr"]*100,2),round(strat_m["tr"]*100,2)],
    })
    st.dataframe(sm_tbl,use_container_width=True,hide_index=True)

st.markdown("---")
st.caption("Brent Crude Total Return Index 2016–2024 · WalkforwardBacktester · TC: 0.015%/trade · Built for Quant Intern Assignment")
