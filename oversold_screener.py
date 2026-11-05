#!/usr/bin/env python3
"""
売られ過ぎ割安株スクリーナー

「今年の株価下落」×「セクター内での割安度(PER/PBR)」×「業績の質」を
組み合わせて、"理由なく売られていそう"な銘柄をスコア順に抽出します。

使い方:
  pip install yfinance pandas numpy
  python oversold_screener.py --tickers tickers.txt --top 20
  python oversold_screener.py --demo            # ネット不要の動作確認

tickers.txt は1行1銘柄。日本株は 7203.T のように末尾に .T
米国株は AAPL のようにそのまま。

※ 投資助言ではありません。スコアは候補の絞り込み用で、
  「下がっている理由」は必ず決算資料・ニュースで確認してください。
"""
import argparse
import sys
from datetime import date

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- データ取得
def rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    val = 100 - 100 / (1 + rs)
    return float(val.iloc[-1]) if len(val.dropna()) else np.nan


def fetch_one(yf, ticker: str):
    t = yf.Ticker(ticker)
    start = f"{date.today().year - 1}-12-25"  # 年初基準を取るため少し前から
    hist = t.history(start=start, auto_adjust=True)
    if hist.empty:
        return None
    close = hist["Close"]
    this_year = close[close.index.year == date.today().year]
    if this_year.empty:
        return None
    base = close[close.index.year < date.today().year]
    base_price = base.iloc[-1] if len(base) else this_year.iloc[0]
    info = t.info or {}
    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName") or "",
        "sector": info.get("sector") or "Unknown",
        "price": float(close.iloc[-1]),
        "ytd_return": float(close.iloc[-1] / base_price - 1),
        "from_high": float(close.iloc[-1] / this_year.max() - 1),
        "rsi14": rsi(close),
        "per": info.get("trailingPE"),
        "pbr": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "earnings_growth": info.get("earningsGrowth"),
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margin": info.get("profitMargins"),
        "debt_to_equity": info.get("debtToEquity"),  # %表記
        "div_yield": info.get("dividendYield"),
    }


JPX_URL = ("https://www.jpx.co.jp/markets/statistics-equities/misc/"
           "tvdivq0000001vg2-att/data_j.xlsx")


def load_jpx_all():
    """JPX公式の上場銘柄一覧から、内国株式(ETF・REIT等を除く)を全件取得。
    戻り値: (ティッカー一覧, {ティッカー: 33業種区分}, {ティッカー: 会社名})"""
    try:
        df = pd.read_excel(JPX_URL)  # openpyxl が必要: pip install openpyxl
    except Exception as e:
        sys.exit(f"JPX銘柄一覧を取得できませんでした: {e}\n"
                 "data_j.xlsx を手動で保存して --jpx-file で指定してください")
    return _parse_jpx(df)


def _parse_jpx(df):
    df = df[df["市場・商品区分"].astype(str).str.contains("内国株式")]
    tickers = [f"{str(c).strip()}.T" for c in df["コード"]]
    sectors = dict(zip(tickers, df["33業種区分"].astype(str)))
    names = dict(zip(tickers, df["銘柄名"].astype(str)))
    return tickers, sectors, names


def fetch_all(tickers, workers=8, sector_map=None, name_map=None):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance が必要です: pip install yfinance")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def job(tk):
        for attempt in range(2):  # 1回だけ再試行
            try:
                return fetch_one(yf, tk)
            except Exception:
                if attempt == 1:
                    return None

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, tk): tk for tk in tickers}
        for fu in as_completed(futs):
            done += 1
            r = fu.result()
            if r:
                if sector_map and futs[fu] in sector_map:
                    r["sector"] = sector_map[futs[fu]]  # 33業種で比較
                if name_map and futs[fu] in name_map:
                    r["name"] = name_map[futs[fu]]  # 日本語の会社名
                rows.append(r)
            if done % 100 == 0 or done == len(tickers):
                print(f"{done}/{len(tickers)} 完了(取得成功 {len(rows)})",
                      file=sys.stderr)
    return pd.DataFrame(rows)


def demo_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    sectors = ["Technology", "Industrials", "Financial Services",
               "Consumer Cyclical", "Healthcare"]
    return pd.DataFrame({
        "ticker": [f"{1000 + i}.T" for i in range(n)],
        "name": [f"Demo{i}" for i in range(n)],
        "sector": rng.choice(sectors, n),
        "price": rng.uniform(500, 5000, n),
        "ytd_return": rng.normal(0.0, 0.25, n),
        "from_high": -np.abs(rng.normal(0.15, 0.12, n)),
        "rsi14": rng.uniform(20, 75, n),
        "per": rng.uniform(5, 40, n),
        "pbr": rng.uniform(0.4, 5, n),
        "roe": rng.normal(0.08, 0.06, n),
        "earnings_growth": rng.normal(0.05, 0.25, n),
        "revenue_growth": rng.normal(0.04, 0.10, n),
        "profit_margin": rng.uniform(0.01, 0.25, n),
        "debt_to_equity": rng.uniform(5, 200, n),
        "div_yield": rng.uniform(0, 0.05, n),
    })


# ---------------------------------------------------------------- スコアリング
def pct_in_sector(df, col, ascending=True, min_group=4):
    """セクター内パーセンタイル(0-1)。小さいセクターは全体で代替。
    ascending=True: 値が小さいほど高スコア(PERなど)。"""
    overall = df[col].rank(pct=True, ascending=not ascending)
    by_sec = df.groupby("sector")[col].rank(pct=True, ascending=not ascending)
    size = df.groupby("sector")[col].transform("count")
    return by_sec.where(size >= min_group, overall)


def score(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # PER/PBRは赤字(負値)や欠損を除外 → スコア計算対象外にする
    d.loc[(d["per"] <= 0) | (d["per"] > 200), "per"] = np.nan
    d.loc[d["pbr"] <= 0, "pbr"] = np.nan

    # 1) 売られ過ぎ度: YTD下落・高値からの下落・RSI低さ
    s_drop = d["ytd_return"].rank(pct=True, ascending=False)
    s_high = d["from_high"].rank(pct=True, ascending=False)
    s_rsi = d["rsi14"].rank(pct=True, ascending=False)
    d["s_oversold"] = (0.5 * s_drop + 0.3 * s_high + 0.2 * s_rsi)

    # 2) 割安度: セクター内でPER/PBRが低い
    d["s_cheap"] = (0.5 * pct_in_sector(d, "per", True)
                    + 0.5 * pct_in_sector(d, "pbr", True))

    # 3) 業績の質: ROE高・増益・利益率高・負債低
    d["s_quality"] = (0.30 * pct_in_sector(d, "roe", False)
                      + 0.30 * pct_in_sector(d, "earnings_growth", False)
                      + 0.20 * pct_in_sector(d, "profit_margin", False)
                      + 0.20 * pct_in_sector(d, "debt_to_equity", True))

    # 欠損は中立(0.5)扱い。ただし割安度が計算不能なら候補から外す
    d["s_oversold"] = d["s_oversold"].fillna(0.5)
    d["s_quality"] = d["s_quality"].fillna(0.5)
    d = d.dropna(subset=["s_cheap"])

    d["score"] = 100 * (0.30 * d["s_oversold"]
                        + 0.35 * d["s_cheap"]
                        + 0.35 * d["s_quality"])

    # バリュートラップ警告
    flags = []
    for _, r in d.iterrows():
        f = []
        if pd.notna(r["earnings_growth"]) and r["earnings_growth"] < -0.2:
            f.append("大幅減益")
        if pd.notna(r["revenue_growth"]) and r["revenue_growth"] < -0.05:
            f.append("減収")
        if pd.notna(r["debt_to_equity"]) and r["debt_to_equity"] > 150:
            f.append("高負債")
        if pd.notna(r["roe"]) and r["roe"] < 0.03:
            f.append("低ROE")
        flags.append(",".join(f))
    d["warning"] = flags
    return d.sort_values("score", ascending=False)


def apply_filters(d, args):
    m = d["ytd_return"] <= -args.min_drop
    if args.max_per:
        m &= d["per"].isna() | (d["per"] <= args.max_per)
    if args.max_pbr:
        m &= d["pbr"].isna() | (d["pbr"] <= args.max_pbr)
    if args.exclude_warning:
        m &= d["warning"] == ""
    return d[m]


# ---------------------------------------------------------------- HTML出力
def write_html(result, path, n_total, top=100):
    import html
    import os
    from datetime import datetime, timedelta, timezone
    jst = timezone(timedelta(hours=9))
    now = datetime.now(jst).strftime("%Y-%m-%d %H:%M JST")

    def pct(v):
        return "-" if pd.isna(v) else f"{v * 100:.1f}%"

    def num(v, nd=1):
        return "-" if pd.isna(v) else f"{v:.{nd}f}"

    cards = []
    for i, (_, r) in enumerate(result.head(top).iterrows(), 1):
        code = str(r["ticker"]).replace(".T", "")
        link = f"https://finance.yahoo.co.jp/quote/{r['ticker']}"
        warn = (f'<div class="warn">⚠ {html.escape(r["warning"])}</div>'
                if r["warning"] else "")
        cards.append(f"""
<div class="card">
  <div class="head"><span class="rank">{i}</span>
    <a href="{link}">{html.escape(str(r['name']))} ({html.escape(code)})</a>
    <span class="score">{r['score']:.1f}</span></div>
  <div class="sec">{html.escape(str(r['sector']))}</div>
  <div class="grid">
    <div>今年<b>{pct(r['ytd_return'])}</b></div>
    <div>高値から<b>{pct(r['from_high'])}</b></div>
    <div>PER<b>{num(r['per'])}</b></div>
    <div>PBR<b>{num(r['pbr'], 2)}</b></div>
    <div>ROE<b>{pct(r['roe'])}</b></div>
    <div>増益率<b>{pct(r['earnings_growth'])}</b></div>
  </div>{warn}
</div>""")

    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>売られ過ぎ割安株スクリーニング</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:12px;background:#f4f5f7;color:#222}}
h1{{font-size:18px;margin:4px 0}} .meta{{font-size:12px;color:#666;margin-bottom:10px}}
.card{{background:#fff;border-radius:10px;padding:10px 12px;margin-bottom:8px;box-shadow:0 1px 2px #0002}}
.head{{display:flex;gap:8px;align-items:center;font-size:15px}}
.head a{{flex:1;color:#0b5cd5;text-decoration:none;font-weight:600}}
.rank{{background:#222;color:#fff;border-radius:50%;min-width:22px;height:22px;text-align:center;font-size:12px;line-height:22px}}
.score{{font-weight:700;color:#0a7d33}} .sec{{font-size:12px;color:#777;margin:2px 0 6px 30px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;font-size:12px;color:#555}}
.grid b{{display:block;font-size:14px;color:#111}}
.warn{{margin-top:6px;font-size:12px;color:#b45309}}
.note{{font-size:11px;color:#777;margin-top:14px;line-height:1.5}}
</style></head><body>
<h1>売られ過ぎ割安株スクリーニング</h1>
<div class="meta">更新 {now} ／ 全{n_total}社中 条件該当 {len(result)}社(上位{min(top, len(result))}社を表示)</div>
{''.join(cards) if cards else '<p>条件に該当する銘柄はありませんでした。</p>'}
<div class="note">スコアは「売られ過ぎ度30%・業種内の割安度35%・業績の質35%」の合成(0〜100)。
⚠は減益・減収・高負債・低ROEなど、安い理由がある可能性のある銘柄です。
データはyfinance(Yahoo Finance)由来で、遅延・欠損・誤りがあり得ます。
投資助言ではありません。下落理由を決算資料やニュースで確認のうえ、ご自身で判断してください。</div>
</body></html>"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="売られ過ぎ割安株スクリーナー")
    ap.add_argument("--tickers", help="ティッカー一覧ファイル(1行1銘柄)")
    ap.add_argument("--demo", action="store_true", help="ダミーデータで実行")
    ap.add_argument("--top", type=int, default=20, help="表示件数")
    ap.add_argument("--min-drop", type=float, default=0.10,
                    help="今年の下落率の下限(0.10=10%%以上下落)")
    ap.add_argument("--max-per", type=float, default=20)
    ap.add_argument("--max-pbr", type=float, default=2.0)
    ap.add_argument("--exclude-warning", action="store_true",
                    help="警告フラグ付き銘柄を除外")
    ap.add_argument("--out", default="screening_result.csv")
    ap.add_argument("--all", action="store_true",
                    help="JPX公式一覧から東証など全上場の内国株を対象にする")
    ap.add_argument("--jpx-file", help="手動保存した data_j.xlsx のパス")
    ap.add_argument("--workers", type=int, default=8, help="並列数")
    ap.add_argument("--cache", default="raw_cache.csv",
                    help="取得データの保存先(再実行時に再利用)")
    ap.add_argument("--use-cache", action="store_true",
                    help="取得せず保存済みデータでスコアだけ再計算")
    ap.add_argument("--html", help="スマホ向けHTMLの出力先(例: docs/index.html)")
    args = ap.parse_args()

    import os
    if args.demo:
        raw = demo_data()
    elif args.use_cache and os.path.exists(args.cache):
        raw = pd.read_csv(args.cache)
    elif args.all or args.jpx_file:
        if args.jpx_file:
            tks, smap, nmap = _parse_jpx(pd.read_excel(args.jpx_file))
        else:
            tks, smap, nmap = load_jpx_all()
        print(f"対象 {len(tks)} 銘柄。全件取得には数十分かかります", file=sys.stderr)
        raw = fetch_all(tks, args.workers, smap, nmap)
        raw.to_csv(args.cache, index=False, encoding="utf-8-sig")
    elif args.tickers:
        with open(args.tickers, encoding="utf-8") as f:
            tks = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        raw = fetch_all(tks, args.workers)
        raw.to_csv(args.cache, index=False, encoding="utf-8-sig")
    else:
        ap.error("--all, --tickers, --demo のいずれかを指定してください")

    if raw.empty:
        sys.exit("データを取得できませんでした")

    scored = score(raw)
    result = apply_filters(scored, args)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")
    if args.html:
        write_html(result, args.html, len(raw), top=max(args.top, 50))

    cols = ["ticker", "name", "sector", "ytd_return", "from_high", "per",
            "pbr", "roe", "earnings_growth", "score", "warning"]
    show = result[cols].head(args.top).copy()
    for c in ["ytd_return", "from_high", "roe", "earnings_growth"]:
        show[c] = (show[c] * 100).round(1).astype("object").where(show[c].notna(), "-")
    show["per"] = show["per"].round(1)
    show["pbr"] = show["pbr"].round(2)
    show["score"] = show["score"].round(1)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.to_string(index=False))
    print(f"\n候補 {len(result)} 件 / 全体 {len(raw)} 件 → {args.out} に保存")
    print("※ 下落率・ROE・増益率の単位は%。warning欄の銘柄は下落理由を要確認。")


if __name__ == "__main__":
    main()
