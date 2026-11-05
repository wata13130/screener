#!/usr/bin/env python3
"""
売られ過ぎ割安株スクリーナー

「今年の株価下落」×「セクター内での割安度(PER/PBR)」×「業績の質」を
組み合わせて、"理由なく売られていそう"な銘柄をスコア順に抽出します。

取得の流れ(Yahooのアクセス制限を避けるため2段階):
  1) 全銘柄の株価をまとめて取得し、今年の下落率を計算
  2) 今年下落した銘柄だけ、PER/PBR/ROEなどの業績データを取得

使い方:
  pip install yfinance pandas numpy openpyxl
  python oversold_screener.py --all --html docs/index.html --out docs/result.csv
  python oversold_screener.py --tickers tickers.txt --top 20
  python oversold_screener.py --demo            # ネット不要の動作確認

※ 投資助言ではありません。スコアは候補の絞り込み用で、
  「下がっている理由」は必ず決算資料・ニュースで確認してください。
"""
import argparse
import json
import random
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

STATS = {}       # 取得件数の記録(HTMLに表示)
COOLDOWN = 20    # 失敗した銘柄を再試行する前の待ち時間(秒)


# ---------------------------------------------------------------- データ取得
def rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    val = 100 - 100 / (1 + rs)
    return float(val.iloc[-1]) if len(val.dropna()) else np.nan


def price_metrics(close):
    """終値の系列から、今年の下落率・高値からの下落・RSIを計算"""
    if close is None:
        return None
    close = close.dropna()
    if close.empty:
        return None
    year = date.today().year
    this_year = close[close.index.year == year]
    if this_year.empty:
        return None
    base = close[close.index.year < year]
    base_price = float(base.iloc[-1]) if len(base) else float(this_year.iloc[0])
    last = float(close.iloc[-1])
    step = max(1, len(this_year) // 50)  # 推移グラフ用に間引く
    spark = [round(float(v), 2) for v in this_year.iloc[::step]]
    if (len(this_year) - 1) % step:
        spark.append(round(last, 2))
    return {
        "price": last,
        "base_price": base_price,
        "spark": json.dumps(spark),
        "ytd_return": last / base_price - 1,
        "from_high": last / float(this_year.max()) - 1,
        "rsi14": rsi(close),
    }


def _extract_close(data, tk, single):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if tk in data.columns.get_level_values(0):
                return data[tk]["Close"]
            return None
        return data["Close"] if single else None
    except Exception:
        return None


def fetch_prices(yf, tickers, chunk=100, retries=2):
    """株価をまとめて取得(100銘柄ずつ)。失敗分は待ってから再試行"""
    start = f"{date.today().year - 1}-12-20"
    out, pending = {}, list(tickers)
    for attempt in range(retries + 1):
        failed = []
        for i in range(0, len(pending), chunk):
            part = pending[i:i + chunk]
            try:
                data = yf.download(part, start=start, auto_adjust=True,
                                   group_by="ticker", progress=False,
                                   threads=True)
            except Exception as e:
                print(f"株価取得エラー: {e}", file=sys.stderr)
                failed += part
                time.sleep(COOLDOWN)
                continue
            for tk in part:
                m = price_metrics(_extract_close(data, tk, len(part) == 1))
                if m:
                    out[tk] = m
                else:
                    failed.append(tk)
            print(f"株価 {min(i + chunk, len(pending))}/{len(pending)} "
                  f"(累計取得 {len(out)})", file=sys.stderr)
        pending = failed
        if not pending:
            break
        if attempt < retries:
            time.sleep(COOLDOWN)
    return out


def fetch_info(yf, tk, tries=4):
    """業績データを取得。制限に当たったら待って再試行"""
    for i in range(tries):
        try:
            info = yf.Ticker(tk).info or {}
            if (info.get("trailingPE") is not None
                    or info.get("priceToBook") is not None
                    or info.get("shortName") or info.get("longName")):
                return {
                    "name": info.get("shortName") or info.get("longName") or "",
                    "sector": info.get("sector") or "Unknown",
                    "per": info.get("trailingPE"),
                    "pbr": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "profit_margin": info.get("profitMargins"),
                    "debt_to_equity": info.get("debtToEquity"),  # %表記
                    "dividend_rate": info.get("dividendRate"),   # 年間配当(円)
                    "div_yield_trailing": info.get("trailingAnnualDividendYield"),
                    "payout_ratio": info.get("payoutRatio"),     # 配当性向
                }
        except Exception:
            pass
        time.sleep(min(2 * (2 ** i), COOLDOWN) + random.random())
    return None


def _div_yield(row):
    """配当利回り(0.025=2.5%)。年間配当÷株価で計算し、無ければ直近実績を使う"""
    rate, price = row.get("dividend_rate"), row.get("price")
    try:
        if rate and price and float(rate) > 0:
            return float(rate) / float(price)
    except (TypeError, ValueError):
        pass
    y = row.get("div_yield_trailing")
    try:
        return float(y) if y is not None and float(y) > 0 else None
    except (TypeError, ValueError):
        return None


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


def fetch_all(tickers, workers=3, sector_map=None, name_map=None,
              min_drop=0.10, max_minutes=240, yf=None):
    if yf is None:
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("yfinance が必要です: pip install yfinance")
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    workers = max(1, min(workers, 3))  # 多すぎるとYahooに制限される
    STATS.clear()
    STATS["universe"] = len(tickers)

    # 1) 株価をまとめて取得 → 今年の下落率で絞る
    prices = fetch_prices(yf, tickers)
    dropped = [tk for tk, m in prices.items() if m["ytd_return"] <= -min_drop]
    STATS["priced"] = len(prices)
    STATS["dropped"] = len(dropped)
    print(f"株価取得 {len(prices)}/{len(tickers)} 社、"
          f"今年{min_drop:.0%}以上下落 {len(dropped)} 社", file=sys.stderr)

    # 2) 下落した銘柄だけ業績データを取得(失敗分は最大3周)
    def job(tk):
        if time.time() - t0 > max_minutes * 60:
            return tk, None
        r = fetch_info(yf, tk)
        time.sleep(0.3)
        return tk, r

    infos, pending = {}, list(dropped)
    for rnd in range(3):
        if not pending or time.time() - t0 > max_minutes * 60:
            break
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for tk, r in ex.map(job, pending):
                done += 1
                if r:
                    infos[tk] = r
                if done % 100 == 0:
                    print(f"業績 {done}/{len(pending)} (累計取得 {len(infos)})",
                          file=sys.stderr)
        pending = [tk for tk in pending if tk not in infos]
        if pending and rnd < 2:
            time.sleep(COOLDOWN * 3)

    rows = []
    for tk in dropped:
        if tk not in infos:
            continue
        row = {"ticker": tk, **prices[tk], **infos[tk]}
        row["div_yield"] = _div_yield(row)
        if sector_map and sector_map.get(tk):
            row["sector"] = sector_map[tk]  # 33業種で比較
        if name_map and name_map.get(tk):
            row["name"] = name_map[tk]      # 日本語の会社名
        rows.append(row)
    STATS["fundamentals"] = len(rows)
    print(f"業績取得 {len(rows)}/{len(dropped)} 社", file=sys.stderr)
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
        "payout_ratio": rng.uniform(0, 1.2, n),
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
WARN_TEXT = {
    "大幅減益": "利益が前年から2割超減っています。",
    "減収": "売上が前年から5%超減っています。",
    "高負債": "負債が自己資本の1.5倍を超えています。",
    "低ROE": "ROE(自己資本利益率)が3%未満で、稼ぐ力が弱めです。",
}

PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>売られ過ぎ割安株スクリーニング</title>
<style>
body{font-family:-apple-system,sans-serif;margin:0;padding:12px;background:#f4f5f7;color:#222}
h1{font-size:18px;margin:4px 0} .meta{font-size:12px;color:#666;margin-bottom:10px;line-height:1.5}
.card{background:#fff;border-radius:10px;padding:10px 12px;margin-bottom:8px;box-shadow:0 1px 2px #0002;cursor:pointer}
.card:active{background:#eef3fb}
.head{display:flex;gap:8px;align-items:center;font-size:15px}
.nm{flex:1;color:#0b5cd5;font-weight:600}
.rank{background:#222;color:#fff;border-radius:50%;min-width:22px;height:22px;text-align:center;font-size:12px;line-height:22px}
.score{font-weight:700;color:#0a7d33} .sec{font-size:12px;color:#777;margin:2px 0 6px 30px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;font-size:12px;color:#555}
.grid b{display:block;font-size:14px;color:#111}
.warn{margin-top:6px;font-size:12px;color:#b45309}
.note{font-size:11px;color:#777;margin-top:14px;line-height:1.5}
#ov{position:fixed;left:0;top:0;right:0;bottom:0;background:#f4f5f7;overflow:auto;display:none;z-index:10}
#ov.on{display:block}
.dh{position:sticky;top:0;background:#fff;padding:10px 12px;border-bottom:1px solid #ddd}
.dh button{border:0;background:none;font-size:15px;color:#0b5cd5;padding:4px 0}
.dc{padding:12px}
.dc h2{font-size:20px;margin:0 0 2px} .dc .code{font-weight:400;color:#777;font-size:15px}
.dc .sec{margin:0 0 8px}
.big{display:flex;justify-content:space-between;align-items:baseline;font-size:22px;font-weight:700;margin-bottom:6px}
.big .score{font-size:16px}
.dc h3{font-size:14px;margin:18px 0 6px;color:#444}
.ch{width:100%;height:auto;background:#fff;border-radius:8px}
.cap{font-size:11px;color:#888;margin-top:2px}
.bar{display:flex;align-items:center;gap:8px;font-size:13px;margin:6px 0}
.bar span{width:84px;color:#555}.bar b{width:30px;text-align:right}
.tr{flex:1;height:8px;background:#e3e6ea;border-radius:4px;overflow:hidden}
.fl{height:100%;background:#0a7d33}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;font-size:14px}
th{text-align:left;font-weight:400;color:#666;padding:8px 12px;border-bottom:1px solid #eee}
td{text-align:right;font-weight:600;padding:8px 12px;border-bottom:1px solid #eee}
.wn{background:#fff7ed;border-radius:8px;padding:8px 12px;margin:6px 0;font-size:13px;color:#9a3412}
.lk{display:block;background:#fff;border-radius:8px;padding:12px;margin:6px 0;color:#0b5cd5;text-decoration:none;font-size:14px}
</style></head><body>
<h1>売られ過ぎ割安株スクリーニング</h1>
<div class="meta">更新 __NOW__<br>__META__</div>
<div id="list">__CARDS__</div>
<div class="note">__NOTE__</div>
<div id="ov"></div>
<script type="application/json" id="d">__DATA__</script>
<script>
var D=JSON.parse(document.getElementById('d').textContent);
var ov=document.getElementById('ov');
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];});}
function chart(x){
  var s=x.spark; if(!s||s.length<2) return '';
  var base=(x.base==null)?s[0]:x.base;
  var W=300,H=110,P=6,all=s.concat([base]);
  var mn=Math.min.apply(null,all),mx=Math.max.apply(null,all);
  if(mx===mn) mx=mn+1;
  function X(k){return P+(W-2*P)*k/(s.length-1);}
  function Y(v){return H-P-(H-2*P)*(v-mn)/(mx-mn);}
  var pts=s.map(function(v,k){return X(k).toFixed(1)+','+Y(v).toFixed(1);}).join(' ');
  var by=Y(base).toFixed(1);
  return '<h3>今年の株価の推移</h3><svg viewBox="0 0 '+W+' '+H+'" class="ch">'
   +'<line x1="'+P+'" x2="'+(W-P)+'" y1="'+by+'" y2="'+by+'" stroke="#999" stroke-dasharray="3 3"/>'
   +'<polyline fill="none" stroke="#0b5cd5" stroke-width="2" points="'+pts+'"/></svg>'
   +'<div class="cap">点線は年初の株価です</div>';
}
function render(i){
  var x=D[i],h='';
  h+='<div class="dh"><button id="bk">← 一覧に戻る</button></div><div class="dc">';
  h+='<h2>'+esc(x.name)+' <span class="code">('+esc(x.code)+')</span></h2>';
  h+='<div class="sec">'+esc(x.sector)+'</div>';
  h+='<div class="big"><span>'+esc(x.price)+'</span><span class="score">スコア '+esc(x.score)+'</span></div>';
  h+=chart(x);
  h+='<h3>スコアの内訳(0〜100)</h3>';
  x.bars.forEach(function(b){
    h+='<div class="bar"><span>'+esc(b[0])+'</span><div class="tr"><div class="fl" style="width:'+Number(b[1])+'%"></div></div><b>'+esc(b[1])+'</b></div>';
  });
  h+='<h3>指標</h3><table>';
  x.rows.forEach(function(r){h+='<tr><th>'+esc(r[0])+'</th><td>'+esc(r[1])+'</td></tr>';});
  h+='</table>';
  if(x.warns.length){
    h+='<h3>注意点</h3>';
    x.warns.forEach(function(w){h+='<div class="wn">⚠ <b>'+esc(w[0])+'</b> '+esc(w[1])+'</div>';});
  }
  h+='<h3>外部サイトで詳しく見る</h3>';
  x.links.forEach(function(l){
    h+='<a class="lk" target="_blank" rel="noopener" href="'+esc(l[1])+'">'+esc(l[0])+' ↗</a>';
  });
  h+='</div>';
  ov.innerHTML=h; ov.scrollTop=0;
  document.getElementById('bk').onclick=function(){history.back();};
}
function openDetail(i){
  render(i); ov.classList.add('on'); document.body.style.overflow='hidden';
  history.pushState({d:i},'','#'+i);
}
function closeDetail(){ov.classList.remove('on');document.body.style.overflow='';}
window.addEventListener('popstate',closeDetail);
document.getElementById('list').addEventListener('click',function(e){
  var c=e.target.closest('.card'); if(c) openDetail(Number(c.getAttribute('data-i')));
});
</script>
</body></html>"""


def _pct(v):
    return "-" if v is None or pd.isna(v) else f"{v * 100:.1f}%"


def _num(v, nd=1):
    return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}"


def _spark_list(r):
    import json
    s = r.get("spark") if hasattr(r, "get") else None
    if isinstance(s, str):
        try:
            s = json.loads(s)
        except Exception:
            s = None
    return s if isinstance(s, list) else []


def build_detail(r):
    """詳細画面用のデータ(表示用に整形済み)"""
    code = str(r["ticker"]).replace(".T", "")
    base = r.get("base_price")
    warns = []
    for w in str(r["warning"]).split(","):
        if w in WARN_TEXT:
            warns.append([w, WARN_TEXT[w]])
    de = r.get("debt_to_equity")
    return {
        "name": str(r["name"]),
        "code": code,
        "sector": str(r["sector"]),
        "score": f"{r['score']:.1f}",
        "price": f"{r['price']:,.0f}円",
        "base": None if base is None or pd.isna(base) else float(base),
        "spark": _spark_list(r),
        "bars": [
            ["売られ過ぎ度", round(float(r["s_oversold"]) * 100)],
            ["割安度", round(float(r["s_cheap"]) * 100)],
            ["業績の質", round(float(r["s_quality"]) * 100)],
        ],
        "rows": [
            ["今年の騰落", _pct(r["ytd_return"])],
            ["今年の高値から", _pct(r["from_high"])],
            ["RSI(14日)", _num(r["rsi14"])],
            ["PER", _num(r["per"])],
            ["PBR", _num(r["pbr"], 2)],
            ["ROE", _pct(r["roe"])],
            ["増益率", _pct(r["earnings_growth"])],
            ["増収率", _pct(r["revenue_growth"])],
            ["利益率", _pct(r["profit_margin"])],
            ["配当利回り", _pct(r.get("div_yield"))],
            ["配当性向", _pct(r.get("payout_ratio"))],
            ["負債資本比率", "-" if de is None or pd.isna(de) else f"{de:.0f}%"],
        ],
        "warns": warns,
        "links": [
            ["Yahoo!ファイナンス", f"https://finance.yahoo.co.jp/quote/{r['ticker']}"],
            ["株探(かぶたん)", f"https://kabutan.jp/stock/?code={code}"],
        ],
    }


def write_html(result, path, n_total, top=100):
    import html
    import json
    import os
    from datetime import datetime, timedelta, timezone
    jst = timezone(timedelta(hours=9))
    now = datetime.now(jst).strftime("%Y-%m-%d %H:%M JST")

    shown = min(top, len(result))
    if STATS:
        meta = (f"対象{STATS['universe']}社 → 株価取得{STATS['priced']}社 → "
                f"今年下落{STATS['dropped']}社 → 業績取得{STATS['fundamentals']}社 "
                f"→ 条件該当{len(result)}社(上位{shown}社を表示)")
        low = (STATS["priced"] < 0.8 * STATS["universe"]
               or STATS["fundamentals"] < 0.8 * max(STATS["dropped"], 1))
        if low:
            meta += ('<div class="warn">⚠ データ取得率が低く、一部の銘柄が'
                     '抜けています</div>')
    else:
        meta = f"全{n_total}社中 条件該当 {len(result)}社(上位{shown}社を表示)"

    cards, details = [], []
    for i, (_, r) in enumerate(result.head(top).iterrows()):
        code = str(r["ticker"]).replace(".T", "")
        warn = (f'<div class="warn">⚠ {html.escape(r["warning"])}</div>'
                if r["warning"] else "")
        cards.append(f"""
<div class="card" data-i="{i}">
  <div class="head"><span class="rank">{i + 1}</span>
    <span class="nm">{html.escape(str(r['name']))} ({html.escape(code)}) ›</span>
    <span class="score">{r['score']:.1f}</span></div>
  <div class="sec">{html.escape(str(r['sector']))}</div>
  <div class="grid">
    <div>今年<b>{_pct(r['ytd_return'])}</b></div>
    <div>高値から<b>{_pct(r['from_high'])}</b></div>
    <div>PER<b>{_num(r['per'])}</b></div>
    <div>PBR<b>{_num(r['pbr'], 2)}</b></div>
    <div>ROE<b>{_pct(r['roe'])}</b></div>
    <div>増益率<b>{_pct(r['earnings_growth'])}</b></div>
    <div>配当利回り<b>{_pct(r.get('div_yield'))}</b></div>
    <div>配当性向<b>{_pct(r.get('payout_ratio'))}</b></div>
  </div>{warn}
</div>""")
        details.append(build_detail(r))

    note = ("スコアは「売られ過ぎ度30%・業種内の割安度35%・業績の質35%」の合成"
            "(0〜100)で、今年下落した銘柄どうしの相対評価です。"
            "⚠は減益・減収・高負債・低ROEなど、安い理由がある可能性のある銘柄です。"
            "株価は前営業日の終値までで、データはyfinance(Yahoo Finance)由来のため、"
            "遅延・欠損・誤りがあり得ます。"
            "投資助言ではありません。下落理由を決算資料やニュースで確認のうえ、"
            "ご自身で判断してください。")
    data = json.dumps(details, ensure_ascii=False).replace("</", "<\\/")
    page = (PAGE.replace("__NOW__", now).replace("__META__", meta)
            .replace("__CARDS__", ''.join(cards) if cards else
                     "<p>条件に該当する銘柄はありませんでした。</p>")
            .replace("__NOTE__", note).replace("__DATA__", data))
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
    ap.add_argument("--workers", type=int, default=3, help="並列数(最大3)")
    ap.add_argument("--max-minutes", type=int, default=240,
                    help="業績取得に使う時間の上限(分)")
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
        print(f"対象 {len(tks)} 銘柄", file=sys.stderr)
        raw = fetch_all(tks, args.workers, smap, nmap,
                        args.min_drop, args.max_minutes)
        raw.to_csv(args.cache, index=False, encoding="utf-8-sig")
    elif args.tickers:
        with open(args.tickers, encoding="utf-8") as f:
            tks = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        raw = fetch_all(tks, args.workers, None, None,
                        args.min_drop, args.max_minutes)
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
    print(f"\n候補 {len(result)} 件 / 業績取得 {len(raw)} 件 → {args.out} に保存")
    if STATS:
        print(f"取得状況: {STATS}")
    print("※ 下落率・ROE・増益率の単位は%。warning欄の銘柄は下落理由を要確認。")


if __name__ == "__main__":
    main()
