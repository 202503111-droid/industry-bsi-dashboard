"""업종별 기업 체감경기 대시보드 (Streamlit)

실행:  streamlit run app.py
데이터: data/bsi_industry.csv  (한국은행 「기업경기조사」, KOSIS 업종별 기업경기실사지수)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="업종별 체감경기", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "bsi_industry.csv"
DATA_DATE = "2026-11-16"  # 자료를 내려받은 날
OLD, NEW = "구분류(~2009.07)", "신분류(2009.08~)"
SYSTEMS = {"신분류 (2009.08~)": NEW, "구분류 (~2009.07)": OLD}
MAX_SELECT = 8
MIN_MONTHS = 12      # 표준편차·회귀를 계산할 최소 개월 수
BLOCK = 12           # 순위 안정성: 다시 뽑는 단위(개월). 1년을 통째로 담아 계절 패턴을 유지
BOOT_N = 2000        # 순위 안정성: 다시 뽑는 횟수
SMOOTH_MONTHS = 3    # 낙폭 이동평균 옵션·회복 판정에 쓰는 개월 수
COVID = ("2020-02", "2021-06")   # 한계 문구에서 '이 기간을 빼면 1위가 바뀌는가'를 볼 구간
NOISE_RATIO = 1.5    # 1위 업종의 월간 변화폭이 중앙값의 이 배수를 넘으면 설문 잡음 경고
SENS_WEAK_R2 = 0.3   # 이 값 미만이면 '경기와의 연동이 약하다'고 설명 (설명용 기준)

# 지표: 체감 업황 = 기업이 느끼는 전반적 경기 (지수 = 100 + 「좋음」 응답 비중% − 「나쁨」 응답 비중%)
INDICATOR = "업황"
LABEL = "체감 업황"

ORDER = [
    "전산업", "제조업", "비제조업",
    "전자·영상·통신장비", "자동차", "조선·기타운수", "1차금속",
    "화학물질·제품", "섬유", "식료품",
    "건설업", "도소매업", "숙박업",
]
# 색: 검증된 범주형 8색(파랑·주황·청록·노랑·분홍·초록·보라·빨강). 인접 색 쌍의 색각이상 구분과 명도 범위를 통과한 순서이며,
# 8개를 넘겨 쓰지 않는다(MAX_SELECT = 8). 추이 차트는 선택한 순서대로 앞에서부터 배정한다.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GROUP_COLORS = {"업종": "#2a78d6", "합계": "#9aa0a8"}        # 업종 = 대표 파랑, 합계 = 무채색(참고용)
NAVY = "#14284b"      # 제목·강조 글자
INK = "#3a4150"       # 차트 글자
GRID = "#e8ebf0"      # 격자선(은은하게)
FONT = "Pretendard, 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"


def md(text: str) -> str:
    """마크다운 문장용: ~ 두 개 사이의 글자에 취소선이 생기지 않도록 ~를 이스케이프."""
    return text.replace("~", "\\~")


def ym(s: str) -> str:
    """'2020-02' → '2020.02'"""
    return s.replace("-", ".")


# ------------------------------------------------------------ 데이터/계산
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["월"] = pd.to_datetime(df["월"], format="%Y-%m")
    df = df.dropna(subset=["값"])
    return df[df["구분"] != "기업구분"]  # 수출·내수 기업구분은 이 앱의 질문(업종 비교)과 무관해 제외


def compute_volatility(f: pd.DataFrame, group_of: dict) -> pd.DataFrame:
    """업종별 표준편차(와 월간 변화폭). 개월 수가 MIN_MONTHS 미만이면 비워서 돌려준다."""
    g = (f.groupby("업종")["값"]
         .agg(표준편차="std", 평균="mean", 개월수="count").reset_index())
    noise = f.sort_values("월").groupby("업종")["값"].apply(lambda s: s.diff().std())
    g["월변화"] = g["업종"].map(noise)  # 한 달 사이 변화의 표준편차: 설문 잡음의 크기를 가늠하는 값
    g = g[g["개월수"] >= MIN_MONTHS].copy()
    g["구분"] = g["업종"].map(group_of)
    return g


def top_without(f: pd.DataFrame, window: tuple):
    """window 구간을 뺀 뒤의 변동성 1위 업종. 선택 기간이 구간과 겹치지 않거나 남는 개월이 부족하면 None."""
    a, b = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    ind = f[f["구분"] == "업종"]
    inside = (ind["월"] >= a) & (ind["월"] <= b)
    if not inside.any():
        return None
    g = ind[~inside].groupby("업종")["값"].agg(["std", "count"])
    g = g[g["count"] >= MIN_MONTHS]
    return None if g.empty else g["std"].idxmax()


@st.cache_data
def rank_stability(f: pd.DataFrame, top: str, n_boot: int = BOOT_N, block: int = BLOCK, seed: int = 0):
    """월별 자료를 block개월 단위로 다시 뽑아(블록 부트스트랩) 표준편차를 구했을 때,
    처음의 1위 업종(top)이 계속 1위로 나오는 비율. 계산할 수 없으면 None."""
    W = f[f["구분"] == "업종"].pivot(index="월", columns="업종", values="값").sort_index()
    A = W.to_numpy(float)
    n, m = A.shape
    if n < block or m < 2 or top not in W.columns:
        return None
    rng = np.random.default_rng(seed)  # 시드 고정: 같은 조건이면 항상 같은 결과
    k = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    sd = A[idx].std(axis=1, ddof=1)  # (n_boot, 업종 수)
    return float((sd.argmax(axis=1) == list(W.columns).index(top)).mean())


def sens_stability(f: pd.DataFrame, group_of: dict, top: str, n_boot: int = BOOT_N,
                   block: int = BLOCK, seed: int = 0):
    """경기 민감도 1위의 안정성. 월별 자료를 block개월 단위로 다시 뽑아(블록 부트스트랩, 업종과 전산업은 같은 달끼리)
    민감도(기울기)를 다시 계산했을 때, 처음의 1위(top)가 계속 1위인 비율과 가장 자주 1위를 뺏은 업종.
    계산할 수 없으면 None."""
    W = f.pivot(index="월", columns="업종", values="값").sort_index()
    if "전산업" not in W.columns:
        return None
    ind = [c for c in W.columns if group_of.get(c) == "업종"]
    W = W[["전산업"] + ind].dropna()
    n = len(W)
    if n < max(block, MIN_MONTHS) or len(ind) < 2 or top not in ind:
        return None
    X, Y = W["전산업"].to_numpy(float), W[ind].to_numpy(float)
    rng = np.random.default_rng(seed)  # 시드 고정
    k = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    x, y = X[idx], Y[idx]                                  # (n_boot, n), (n_boot, n, 업종 수)
    xc = x - x.mean(axis=1, keepdims=True)
    slope = (xc[:, :, None] * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / (xc ** 2).sum(axis=1)[:, None]
    win = slope.argmax(axis=1)
    share = np.bincount(win, minlength=len(ind)) / n_boot
    ti = ind.index(top)
    others = [(share[i], ind[i]) for i in range(len(ind)) if i != ti]
    c_share, c_name = max(others)
    return {"p": float(share[ti]), "challenger": c_name, "challenger_p": float(c_share)}


def compute_drawdown(f: pd.DataFrame, group_of: dict, smooth: bool = False) -> pd.DataFrame:
    """선택 기간 안에서 최고점에서 이후 최저점까지의 최대 하락폭(포인트).
    smooth=True면 SMOOTH_MONTHS개월 이동평균 위에서 계산해 한두 달 튀는 값의 영향을 줄인다."""
    rnd = (lambda x: round(float(x), 1)) if smooth else (lambda x: int(round(x)))
    rows = []
    for name, g in f.groupby("업종"):
        g = g.sort_values("월")
        v = g["값"].to_numpy(dtype=float)
        months = g["월"].dt.strftime("%Y-%m").to_numpy()
        if smooth:
            v = pd.Series(v).rolling(SMOOTH_MONTHS).mean().dropna().to_numpy()
            months = months[SMOOTH_MONTHS - 1:]
        if len(v) < 2:
            continue
        drop = np.maximum.accumulate(v) - v
        trough = int(drop.argmax())            # 가장 크게 떨어진 시점
        peak = int(np.argmax(v[:trough + 1]))  # 그 직전까지의 최고점
        rows.append({
            "업종": name, "구분": group_of.get(name),
            "최고": rnd(v[peak]), "최고월": months[peak],
            "최저": rnd(v[trough]), "최저월": months[trough],
            "낙폭": float(drop[trough]),
        })
    return pd.DataFrame(rows)


def compute_sensitivity(f: pd.DataFrame, group_of: dict) -> pd.DataFrame:
    """업종별 경기 민감도: 업종 지수를 전산업 지수에 회귀한 기울기, 설명력(R²), 설명 안 되는 변동(잔차 표준편차).
    전산업이 없거나 개월 수가 부족하면 빈 표."""
    W = f.pivot(index="월", columns="업종", values="값").sort_index()
    if "전산업" not in W.columns:
        return pd.DataFrame()
    rows = []
    for name in W.columns:
        if group_of.get(name) != "업종":
            continue
        d = W[["전산업", name]].dropna()
        if len(d) < MIN_MONTHS:
            continue
        x, y = d["전산업"].to_numpy(float), d[name].to_numpy(float)
        slope, icpt = np.polyfit(x, y, 1)
        resid = y - (icpt + slope * x)
        rows.append({
            "업종": name, "구분": "업종", "민감도": float(slope),
            "R2": float(np.corrcoef(x, y)[0, 1] ** 2),
            "표준편차": float(y.std(ddof=1)), "고유변동": float(resid.std(ddof=1)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 결론 문장
def vol_stats(g: pd.DataFrame):
    """변동성 표에서 업종 순위·평균 등을 뽑는다. 업종이 없으면 None."""
    g_ind = g[g["구분"] == "업종"]
    if g_ind.empty:
        return None
    ranked = g_ind.sort_values("표준편차", ascending=False)
    return {
        "ind": g_ind, "ref": g[g["구분"] != "업종"], "ranked": ranked,
        "avg": float(g_ind["표준편차"].mean()),
        "top": ranked.iloc[0], "bottom": ranked.iloc[-1],
        "noise": g_ind.set_index("업종")["월변화"],
    }


def vol_sentences(v: dict, label: str) -> list:
    top, bottom, avg, ranked = v["top"], v["bottom"], v["avg"], v["ranked"]
    text = (
        f"**출렁임** · {label} 지수가 가장 크게 출렁인 업종은 **{top['업종']}** "
        f"(표준편차 {top['표준편차']:.1f})입니다. 업종 평균 {avg:.1f}의 "
        f"{top['표준편차'] / avg:.1f}배 수준이며, 상위 3개는 "
        f"{' · '.join(ranked.head(3)['업종'])}입니다. 가장 안정적인 업종은 "
        f"**{bottom['업종']}** ({bottom['표준편차']:.1f})입니다."
    )
    p = v.get("stability")
    if p is not None and p < 0.6:  # 순위가 불안정하면 결론 문장 자체에서 단정을 피한다
        text += (f" 다만 월별 자료를 다시 뽑으면 1위가 유지될 확률이 {p * 100:.0f}%뿐이라 "
                 "1위는 잠정적입니다.")
    return [text]


def dd_sentences(dd_ind: pd.DataFrame, label: str, smooth: bool = False) -> list:
    ranked = dd_ind.sort_values("낙폭", ascending=False)
    top, bottom = ranked.iloc[0], ranked.iloc[-1]
    avg = float(dd_ind["낙폭"].mean())
    note = f" ({SMOOTH_MONTHS}개월 이동평균 기준)" if smooth else ""
    out = [
        f"**하락폭**{note} · {label} 지수가 가장 크게 떨어진 업종은 **{top['업종']}** "
        f"({top['최고월']} {top['최고']} → {top['최저월']} {top['최저']}, "
        f"**{top['낙폭']:.0f}p 하락**)입니다. 평균 {avg:.1f}p의 "
        f"{top['낙폭'] / avg:.1f}배 수준이며, 가장 덜 떨어진 업종은 "
        f"**{bottom['업종']}** ({bottom['낙폭']:.0f}p)입니다."
    ]
    return out


def weak_links(sens: pd.DataFrame) -> list:
    return [] if sens.empty else sens[sens["R2"] < SENS_WEAK_R2]["업종"].tolist()


def sens_sentences(sens: pd.DataFrame, stab: dict = None) -> list:
    r = sens.sort_values("민감도", ascending=False)
    top, bottom = r.iloc[0], r.iloc[-1]
    text = (
        f"**경기 민감도** · 전산업 지수가 1p 움직일 때 가장 크게 따라 움직인 업종은 **{top['업종']}** "
        f"({top['민감도']:.2f}p, 전산업으로 설명되는 비중 R² {top['R2']:.2f})입니다. "
        f"가장 둔감한 업종은 **{bottom['업종']}** ({bottom['민감도']:.2f}p)입니다."
    )
    weak = weak_links(sens)
    if weak:
        text += (f" 전산업과의 연동이 약한 업종(R² {SENS_WEAK_R2:.1f} 미만)은 {' · '.join(weak)}이며, "
                 "이들의 등락은 경기보다 업종 고유 요인의 몫이 큽니다.")
    if stab is not None and stab["p"] < 0.6:  # 순위가 불안정하면 단정을 피한다
        text += (f" 다만 월별 자료를 다시 뽑으면 1위가 유지될 확률이 {stab['p'] * 100:.0f}%뿐이고 "
                 f"가장 자주 1위를 차지한 업종은 {stab['challenger']}({stab['challenger_p'] * 100:.0f}%)여서, 1위는 잠정적입니다.")
    return [text]


def limit_lines(v: dict, sens: pd.DataFrame = None, stab: dict = None) -> list:
    """한계 문장. 현재 데이터에서 계산한 숫자로 설명한다."""
    avg = v["avg"]
    top = v["top"]["업종"]
    lines = []
    if stab is not None and sens is not None and not sens.empty:
        s_name = sens.sort_values("민감도", ascending=False).iloc[0]["업종"]
        q = stab["p"]
        head = (f"월별 자료를 {BLOCK}개월 단위로 {BOOT_N:,}번 다시 뽑아 민감도를 다시 계산하면, "
                f"민감도 1위({s_name})가 1위로 유지될 확률은 {q * 100:.0f}%이고 "
                f"가장 자주 1위를 차지한 업종은 {stab['challenger']}({stab['challenger_p'] * 100:.0f}%)입니다. ")
        if q >= 0.8:
            lines.append(head + "순위가 안정적입니다.")
        elif q >= 0.6:
            lines.append(head + "순위의 안정성은 보통 수준이어서 1위와 2위권의 차이를 단정하기 어렵습니다.")
        else:
            lines.append(head + "기간을 조금만 바꿔도 1위가 바뀔 수 있습니다.")
    p = v.get("stability")
    if p is not None:
        head = (f"월별 자료를 {BLOCK}개월 단위로 {BOOT_N:,}번 다시 뽑아 계산하면, "
                f"표준편차 1위({top})가 1위로 유지될 확률은 {p * 100:.0f}%입니다. ")
        if p >= 0.8:
            lines.append(head + "순위가 안정적입니다.")
        elif p >= 0.6:
            lines.append(head + "순위의 안정성은 보통 수준입니다.")
        else:
            lines.append(head + "기간을 조금만 바꿔도 순위가 바뀔 수 있습니다.")
    alt = v.get("covid_top")  # 선택 기간이 코로나 구간을 포함할 때만 값이 있다
    if alt is not None:
        span = f"{ym(COVID[0])}~{ym(COVID[1])}"
        if alt != top:
            lines.append(f"코로나 기간({span})을 빼고 다시 계산하면 변동성 1위가 {top}에서 {alt}(으)로 바뀝니다. "
                         "1위가 한 번의 큰 사건에 기댄 결과일 수 있습니다.")
        else:
            lines.append(f"코로나 기간({span})을 빼고 다시 계산해도 변동성 1위는 {top}으로 같습니다.")
    nz = v["noise"]
    med, t = float(nz.median()), float(nz[top])
    if med > 0 and t > NOISE_RATIO * med:
        lines.append(f"{top}의 월간 변화폭(한 달 사이 변화의 표준편차 {t:.1f})이 업종 중앙값({med:.1f})의 "
                     f"{t / med:.1f}배입니다. 표본이 작은 업종은 설문 잡음이 커서, 표준편차에 그 잡음이 섞였을 수 있습니다.")
    tot = v["ref"]  # 합계(전산업·제조업·비제조업)
    if not tot.empty:
        lo, hi = tot["표준편차"].min() / avg * 100, tot["표준편차"].max() / avg * 100
        lines.append("합계(전산업·제조업·비제조업)는 여러 업종이 합산되며 등락이 상쇄돼 표준편차가 "
                     f"업종 평균({avg:.1f})의 {lo:.0f}~{hi:.0f}% 수준입니다. "
                     "그래서 합계를 업종과 직접 비교하는 것은 큰 의미가 없습니다.")
    if sens is not None and not sens.empty:
        s_top = sens.sort_values("민감도", ascending=False).iloc[0]
        lines.append(f"경기 민감도 1위({s_top['업종']})는 등락의 {s_top['R2'] * 100:.0f}%(R² {s_top['R2']:.2f})가 전산업 흐름으로 "
                     "설명되고 나머지는 업종 고유 요인입니다. 민감도는 함께 움직인 정도이지 원인이 아니며, 전산업에 이 업종도 "
                     "일부 포함돼 있어 연동이 다소 크게 나올 수 있습니다.")
    lines.append("지수는 기업인의 체감(설문)이며 실제 매출액·이익이 아닙니다. 표준편차와 낙폭에는 "
                 "경기 외의 업종 고유 요인도 섞여 있고, 신분류와 구분류는 값을 직접 비교할 수 없습니다.")
    return lines


# ------------------------------------------------------------------ 차트
def style(fig: go.Figure) -> go.Figure:
    """모든 차트에 공통으로 적용하는 모양: 흰 바탕, 은은한 격자, 차분한 글자색."""
    fig.update_layout(
        template="plotly_white", font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="white", plot_bgcolor="white",
        hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=GRID),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def ordered_names(g: pd.DataFrame, value_col: str) -> list:
    """막대 순서: 구분(업종 → 합계) 순, 같은 구분 안에서는 값이 큰 순."""
    rank = {"업종": 0, "합계": 1}
    t = g.assign(_r=g["구분"].map(rank)).sort_values(["_r", value_col], ascending=[True, False])
    return t["업종"].tolist()


def make_bar_figure(g: pd.DataFrame, value_col: str, ref_value, xaxis_title: str,
                    hover_cols: list, hover: str, text_fmt: str, ref_text: str = None) -> go.Figure:
    """가로 막대 (큰 값이 위). 업종과 합계를 색으로 구분하고, 기준선은 점선."""
    order = ordered_names(g, value_col)
    fig = go.Figure()
    for grp in ("업종", "합계"):
        sub = g[g["구분"] == grp]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub[value_col], y=sub["업종"], orientation="h", name=grp,
            marker_color=GROUP_COLORS[grp],
            customdata=sub[hover_cols].to_numpy(),
            texttemplate=text_fmt, textposition="outside", cliponaxis=False,
            hovertemplate=hover,
        ))
    if ref_value is not None:
        fig.add_vline(x=ref_value, line_dash="dash", line_color="gray",
                      annotation_text=ref_text or f"업종 평균 {ref_value:.1f}",
                      annotation_position="top")
    fig.update_layout(
        height=max(280, 30 * len(order) + 140),
        xaxis_title=xaxis_title,
        yaxis=dict(categoryorder="array", categoryarray=order, autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=30, t=60, b=10),
    )
    return style(fig)


def make_vol_figure(g, ref_value, label):
    return make_bar_figure(
        g, "표준편차", ref_value, f"{label} 지수의 표준편차 (클수록 크게 출렁임)",
        ["평균", "개월수"],
        "<b>%{y}</b><br>표준편차: %{x:.1f}<br>평균: %{customdata[0]:.1f}"
        "<br>개월 수: %{customdata[1]}<extra></extra>",
        "%{x:.1f}")


def make_drawdown_figure(g, ref_value, smooth=False):
    fmt = "%{x:.0f}"
    return make_bar_figure(
        g, "낙폭", ref_value,
        "낙폭 (기간 내 최고점 − 이후 최저점, 포인트" + (f", {SMOOTH_MONTHS}개월 이동평균)" if smooth else ")"),
        ["최고", "최고월", "최저", "최저월"],
        "<b>%{y}</b><br>낙폭: %{x:.0f}p<br>최고: %{customdata[0]} (%{customdata[1]})"
        "<br>최저: %{customdata[2]} (%{customdata[3]})<extra></extra>",
        fmt)


def make_sens_figure(g):
    return make_bar_figure(
        g, "민감도", 1.0, "경기 민감도 (전산업이 1p 움직일 때 업종이 따라 움직인 폭)",
        ["R2", "고유변동", "표준편차"],
        "<b>%{y}</b><br>민감도: %{x:.2f}<br>R²: %{customdata[0]:.2f}"
        "<br>고유 변동: %{customdata[1]:.1f}<br>표준편차: %{customdata[2]:.1f}<extra></extra>",
        "%{x:.2f}", ref_text="전산업 = 1.0")


def make_line_figure(f, selected, colors, labels, label):
    fig = go.Figure()
    for name in selected:
        sub = f[f["업종"] == name].sort_values("월")
        fig.add_trace(go.Scatter(
            x=sub["월"], y=sub["값"], mode="lines", name=labels[name],
            line=dict(color=colors[name], width=2), hovertemplate="%{y}",
        ))
    fig.add_hline(y=100, line_dash="dash", line_color="gray",
                  annotation_text="기준선 100", annotation_position="top left")
    fig.update_layout(
        height=480, hovermode="x unified",
        yaxis_title=f"{label} 지수 (100 기준)",
        xaxis=dict(tickformat="%Y", hoverformat="%Y년 %m월"),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return style(fig)


# ----------------------------------------------------------------- 화면
df = load_data()

names_in_data = set(df["업종"].unique())
options = [n for n in ORDER if n in names_in_data]
options += sorted(names_in_data - set(options))
group_of = df.drop_duplicates("업종").set_index("업종")["구분"].to_dict()
labels = {n: (n if group_of[n] == "업종" else f"{n} ({group_of[n]})") for n in options}


def system_span(system: str) -> str:
    m = df[df["분류체계"] == system]["월"]
    return f"{m.min():%Y.%m}~{m.max():%Y.%m}"


st.markdown(f"""
<style>
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
/* 상단 배너 */
.banner {{ background: linear-gradient(135deg, {NAVY} 0%, #1f3d73 100%); border-radius: 10px;
          padding: 1.5rem 1.8rem 1.4rem; margin-bottom: 1rem; }}
.banner .eyebrow {{ color: #9fb8e0; font-size: .76rem; font-weight: 700; letter-spacing: .1em; margin-bottom: .35rem; }}
.banner .title {{ color: #ffffff; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.25; }}
.banner .lead {{ color: #d6e2f5; font-size: 1.02rem; margin-top: .45rem; }}
h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; font-weight: 700; }}
/* 지표 카드 */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #ffffff; border-color: #dfe4ec; border-radius: 8px; }}
[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {{
    border-top: 3px solid {NAVY}; box-shadow: 0 1px 3px rgba(20,40,75,.08); }}
[data-testid="stMetricLabel"] p {{ color: #52606d; font-weight: 600; font-size: .86rem; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; font-size: 1.9rem; }}
/* 탭 */
button[role="tab"] p {{ font-weight: 600; font-size: .98rem; }}
button[role="tab"][aria-selected="true"] {{ color: {NAVY}; }}
div[data-baseweb="tab-highlight"] {{ background-color: {NAVY}; height: 3px; }}
/* 사이드바 */
[data-testid="stSidebar"] {{ border-right: 1px solid #dfe4ec; }}
[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; color: {NAVY}; border-bottom: 2px solid {NAVY};
                               padding-bottom: .4rem; }}
/* 펼침 상자 */
[data-testid="stExpander"] {{ border-color: #dfe4ec; border-radius: 8px; background: #fbfcfe; }}
</style>
<div class="banner">
<div class="eyebrow">한국은행 기업경기조사 · 업종별 체감 업황</div>
<div class="title">업종별 기업 체감경기 대시보드</div>
<div class="lead">경기 변동에 가장 민감하게 반응한 업종은 어디인가?</div>
</div>
""", unsafe_allow_html=True)
st.caption(md("왼쪽에서 조건을 바꾸면 지표, 차트, 결론이 함께 바뀝니다. "
              f"자료: 한국은행 「기업경기조사」 업종별 기업경기실사지수(국가통계포털 KOSIS), 월별 {df['월'].min():%Y.%m}~{df['월'].max():%Y.%m}. "
              "지수 = 100 + 긍정 응답 비중(%) − 부정 응답 비중(%), 기준값 100. "
              "기업인의 체감을 나타낸 값이며 실제 매출·이익이 아닙니다."))

with st.sidebar:
    st.header("조건 선택")
    scope = st.radio(
        "분류체계", list(SYSTEMS), index=0,
        help="업종 분류가 2009.08에 개편되어 두 구간은 서로 다른 분류체계입니다. 값을 직접 비교하지 마세요.",
    )
    system = SYSTEMS[scope]
    st.caption(md(f"**{LABEL}**: 기업이 느끼는 전반적 경기\n\n"
                  "지수 = 100 + 「좋음」 응답 비중(%) − 「나쁨」 응답 비중(%)"))

    scope_months = [m.strftime("%Y-%m")
                    for m in sorted(df[df["분류체계"] == system]["월"].unique())]
    c1, c2 = st.columns(2)
    start = c1.selectbox("시작 월", scope_months, index=0, key=f"start_{system}")
    end = c2.selectbox("종료 월", scope_months, index=len(scope_months) - 1, key=f"end_{system}")
    smooth = st.checkbox(
        f"낙폭을 {SMOOTH_MONTHS}개월 이동평균으로 계산", value=False,
        help="한두 달 튀는 값이 최고점·최저점이 되는 것을 줄입니다. 낙폭 탭과 결론에 적용됩니다.")

if start >= end:
    st.error("시작 월이 종료 월보다 앞서야 합니다. 기간을 다시 선택해 주세요.")
    st.stop()

label = LABEL
f_all = df[(df["지표"] == INDICATOR) & (df["분류체계"] == system)
           & (df["월"] >= pd.Timestamp(start)) & (df["월"] <= pd.Timestamp(end))]

vol = compute_volatility(f_all, group_of)
v = vol_stats(vol) if not vol.empty else None
if v is not None:
    v["stability"] = rank_stability(f_all, v["top"]["업종"])
    v["covid_top"] = top_without(f_all, COVID)
dd = compute_drawdown(f_all, group_of, smooth)
dd_ind = dd[dd["구분"] == "업종"] if not dd.empty else dd
sens = compute_sensitivity(f_all, group_of)
sens_stab = None
if not sens.empty:
    sens_stab = sens_stability(f_all, group_of, sens.sort_values("민감도", ascending=False).iloc[0]["업종"])

# ---------------------------------------------------------------- 핵심 지표 + 결론 (화면 맨 위)
def kpi(col, title: str, value: str, sub: str, help_text: str):
    with col:
        with st.container(border=True):
            st.metric(title, value, help=help_text)
            st.caption(md(sub))


if v is None or dd_ind.empty:
    st.warning(f"비교할 데이터가 부족합니다. 기간을 {MIN_MONTHS}개월 이상으로 늘려 주세요.")
else:
    a = v["top"]["업종"]
    dd_top = dd_ind.sort_values("낙폭", ascending=False).iloc[0]
    b = dd_top["업종"]
    st.caption(md(f"{system} · {label} · {start} ~ {end}"))
    k1, k2, k3, k4 = st.columns(4)
    if sens.empty:
        kpi(k1, "경기 민감도 1위", "-", "계산할 데이터 부족", "전산업 지수가 1p 움직일 때 따라 움직인 폭")
    else:
        s_top = sens.sort_values("민감도", ascending=False).iloc[0]
        kpi(k1, "경기 민감도 1위", s_top["업종"], f"전산업 1p당 {s_top['민감도']:.2f}p, R² {s_top['R2']:.2f}",
            "전산업 지수가 1p 움직일 때 가장 크게 따라 움직인 업종 (이 앱의 주 질문에 대한 답)")
    kpi(k2, "출렁임 1위", a, f"표준편차 {v['top']['표준편차']:.1f} (업종 평균 {v['avg']:.1f})",
        "기간 중 지수가 가장 크게 오르내린 업종")
    kpi(k3, "낙폭 1위", b, f"{dd_top['최고월']} → {dd_top['최저월']}, {dd_top['낙폭']:.0f}p 하락",
        "기간 안에서 최고점에서 이후 최저점까지 가장 크게 떨어진 업종")
    q = sens_stab["p"] if sens_stab else None
    kpi(k4, "민감도 1위 안정성", "계산 불가" if q is None else f"{q * 100:.0f}%",
        (f"가장 자주 1위를 뺏은 업종: {sens_stab['challenger']} {sens_stab['challenger_p'] * 100:.0f}%"
         if q is not None else "기간이 짧아 계산하지 않음"),
        f"월별 자료를 {BLOCK}개월 단위로 {BOOT_N:,}번 다시 뽑아 민감도를 다시 구했을 때 민감도 1위가 1위로 유지되는 비율. "
        "80% 이상 안정, 60% 미만은 순위가 바뀔 수 있음(설명용 기준)")


tab_sens, tab_trend, tab_vol, tab_dd = st.tabs(
    ["경기 민감도", "추이", "변동성", "낙폭"])

# ---------------------------------------------------------------- 추이
with tab_trend:
    default_sel = []
    if v is not None:
        default_sel = list(v["ranked"].head(3)["업종"]) + [n for n in ["전산업"] if n in options]
    # 키에 왼쪽 선택값을 넣어, 조건이 바뀌면 새 기본값(변동성 상위 3개 + 전산업)으로 다시 시작
    selected = st.multiselect(
        f"업종 (최대 {MAX_SELECT}개, 기본값: 변동성 상위 3개 + 전산업)", options,
        default=default_sel, format_func=lambda n: labels[n], max_selections=MAX_SELECT,
        key=f"sel_{system}_{start}_{end}",
    )
    if not selected:
        st.info("업종을 하나 이상 선택해 주세요.")
    else:
        # 색은 선택한 순서대로 앞에서부터 배정(최대 8개, 같은 색을 돌려 쓰지 않음)
        colors = {n: PALETTE[i] for i, n in enumerate(selected)}
        st.plotly_chart(make_line_figure(f_all[f_all["업종"].isin(selected)],
                                         selected, colors, labels, label))
        st.caption("100보다 높으면 긍정 응답이 부정 응답보다 많다는 뜻입니다. 업황은 평상시에도 "
                   "100 미만인 달이 많으므로, 수준보다 시기 간 변화로 읽으세요.")

# ---------------------------------------------------------------- 변동성
with tab_vol:
    st.markdown(md(f"**{label} 지수의 표준편차**가 클수록 기간 중 크게 출렁였다는 뜻입니다. "
                   "다만 이 값에는 경기와 무관한 업종 고유의 등락도 섞여 있으니, 경기에 얼마나 연동됐는지는 "
                   "‘경기 민감도’ 탭에서 함께 보세요. "
                   "기준선(업종 평균)은 업종 항목만으로 계산하고, 합계(전산업·제조업·비제조업)는 참고용입니다."))
    if v is None:
        st.warning(f"비교할 데이터가 없습니다. 기간을 {MIN_MONTHS}개월 이상으로 늘려 주세요.")
    else:
        st.plotly_chart(make_vol_figure(vol, v["avg"], label))

# ---------------------------------------------------------------- 경기 민감도
with tab_sens:
    st.markdown(md(
        "**경기 민감도** = 전산업 지수가 1p 변할 때 해당 업종 지수가 평균적으로 몇 p 변했는지(회귀 기울기)입니다. "
        "1보다 크면 경기 전반보다 더 크게, 작으면 더 둔하게 반응했다는 뜻입니다. "
        "**R²**는 업종 등락 중 전산업(경기 전반)으로 설명되는 비중이고, 낮을수록 경기보다 업종 고유 요인으로 움직였다는 뜻입니다."))
    if sens.empty:
        st.warning(f"비교할 데이터가 없습니다. 기간을 {MIN_MONTHS}개월 이상으로 늘려 주세요.")
    else:
        st.plotly_chart(make_sens_figure(sens))
        show = (sens.sort_values("민감도", ascending=False)
                [["업종", "민감도", "R2", "표준편차", "고유변동"]]
                .rename(columns={"R2": "R² (경기로 설명되는 비중)", "고유변동": "고유 변동 (잔차 표준편차)"})
                .round(2))
        st.dataframe(show, hide_index=True)
        st.caption(md("전산업에는 각 업종이 일부 포함돼 있어 연동이 다소 크게 나올 수 있습니다. "
                      "기울기는 업종 간 상대 비교용이며, 원인과 결과가 아니라 함께 움직인 정도를 나타냅니다."))

# ---------------------------------------------------------------- 낙폭
with tab_dd:
    st.markdown("**낙폭** = 선택한 기간 안에서 최고점에서 이후 최저점까지 떨어진 최대 폭(포인트)입니다. "
                "클수록 더 크게 떨어졌다는 뜻이며, 기간은 왼쪽에서 고른 기간입니다."
                + (f" 지금은 {SMOOTH_MONTHS}개월 이동평균 위에서 계산했습니다." if smooth else ""))
    if dd_ind.empty:
        st.warning("비교할 데이터가 없습니다. 기간을 늘려 주세요.")
    else:
        st.plotly_chart(make_drawdown_figure(dd, float(dd_ind["낙폭"].mean()), smooth))
        st.caption("합계는 여러 업종이 합쳐져 하락이 상쇄되므로 낙폭이 작게 나옵니다. "
                   "낙폭은 지수 수준이 아니라 변화폭이며, 기간의 시작·종료에 따라 달라집니다.")
        first3 = [pd.Timestamp(m).strftime("%Y-%m") for m in sorted(f_all["월"].unique())[:3]]
        edge = dd_ind[dd_ind["최고월"].isin(first3)]["업종"].tolist()
        if edge:
            st.caption(md(f"{' · '.join(edge)}은(는) 최고점이 선택 기간의 첫 3개월 안에 있습니다. "
                          "기간의 시작점이 그대로 최고점이 된 것일 수 있어, 시작월을 앞당기면 낙폭이 달라질 수 있습니다."))

# ---------------------------------------------------------------- 결론 (탭 아래)
if v is not None and not dd_ind.empty:
    a = v["top"]["업종"]
    b = dd_ind.sort_values("낙폭", ascending=False).iloc[0]["업종"]
    lines = (sens_sentences(sens, sens_stab) if not sens.empty else []) \
        + vol_sentences(v, label) + dd_sentences(dd_ind, label, smooth)
    lines.append(f"변동성 1위({a})와 낙폭 1위({b})가 " + (
        "같습니다." if a == b else
        "다릅니다. 지수가 오르내린 폭(변동성)과 한 번의 큰 추락(낙폭)은 서로 다른 측면입니다."))
    if a in weak_links(sens):
        lines.append(f"변동성 1위({a})는 전산업과의 연동이 약해, 그 출렁임은 경기보다 업종 고유 요인의 몫이 큽니다.")
    st.subheader("결론")
    with st.container(border=True):
        st.markdown(md("\n\n".join(lines)))
    with st.expander("이 결론의 한계", expanded=True):
        for item in limit_lines(v, sens, sens_stab):
            st.markdown(md(f"- {item}"))

with st.expander("출처와 집계 기준"):
    notes = (df.dropna(subset=["비고"]).drop_duplicates(["업종", "비고"])[["업종", "비고"]]
             .itertuples(index=False))
    note_lines = "".join(f"  - {n}: {t}\n" for n, t in notes)
    st.markdown(md(
        "- **출처:** 한국은행 「기업경기조사」, 국가통계포털(KOSIS) 「업종별 기업경기실사지수」\n"
        f"  - 신분류: 통계표 ID DT_512Y007, 월별 {system_span(NEW)}\n"
        f"  - 구분류: 통계표 ID DT_512Y001, 이 앱에서는 월별 {system_span(OLD)} 사용\n"
        f"  - 자료 내려받은 날: {DATA_DATE}\n"
        "- **제외:** 수출·내수 기업구분은 이 앱의 질문(업종 비교)과 무관해 쓰지 않았습니다.\n"
        "- **지수:** 체감 업황 = 100 + 「좋음」 응답 업체 비중(%) − 「나쁨」 응답 업체 비중(%)\n"
        "  - 매출·채산성 지수도 있으나, 질문(전반적 체감경기)에는 업황이 가장 직접적이라 쓰지 않았습니다.\n"
        "- **분류 개편:** 2009.08을 기준으로 업종 분류가 달라져 한 번에 하나의 분류체계만 보여 줍니다. "
        "업종별 포함 범위 차이(자료의 비고):\n" + note_lines +
        "- **경기 민감도:** 업종 지수를 전산업 지수에 단순회귀한 기울기이고, R²는 그 설명력입니다. "
        f"R² {SENS_WEAK_R2:.1f} 미만을 ‘연동이 약하다’고 보는 것은 설명용 기준입니다.\n"
        f"- **순위 안정성:** 월별 자료를 {BLOCK}개월 단위(1년을 통째로 담아 계절 패턴이 유지됨)로 "
        f"{BOOT_N:,}번 다시 뽑아(업종과 전산업은 같은 달끼리) 민감도를 다시 구하고, 처음의 민감도 1위가 계속 1위인 비율을 계산했습니다. "
        "표준편차 1위도 같은 방식으로 계산해 한계 문장에 씁니다(블록 부트스트랩, 난수 시드 고정). "
        "80% 이상 안정적, 60~80% 보통, 60% 미만은 순위가 바뀔 수 있다고 봅니다(설명용 기준).\n"
        f"- **낙폭 이동평균 옵션:** 켜면 {SMOOTH_MONTHS}개월 이동평균 위에서 최고점·최저점을 찾습니다(앞쪽 {SMOOTH_MONTHS - 1}개월은 계산에서 빠짐).\n"
        "- **업종 평균:** 이 앱의 업종 항목 평균이며 전체 경제의 평균이 아닙니다."
    ))
