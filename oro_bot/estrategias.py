"""Entradas estratégicas en movimientos grandes del oro (XAUUSD 4H).

Reglas de riesgo del reel en todas: arriesgar 1% de la cuenta por operación
y take profit 1:4. Cada estrategia define su propia entrada y distancia de stop.

Para no sobreajustar, las reglas se prueban primero en 2012-2022 (datos que no
se usaron para elegirlas) y después en el último año.
"""
import pandas as pd

SPREAD = 0.35   # USD por onza, costo de ida y vuelta
RISK = 0.01     # 1% de la cuenta por operación
RR = 4          # take profit 1:4


def load(path, scale=1):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "datetime"})
    df["datetime"] = pd.to_datetime(df.datetime)
    df = df.set_index("datetime")[["open", "high", "low", "close"]] / scale
    return indicators(df)


def indicators(df):
    c = df.close
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    d = c.diff()
    rs = d.clip(lower=0).ewm(alpha=1 / 14).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    df["rsi"] = 100 - 100 / (1 + rs)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_up"], df["bb_dn"] = mid + 2 * sd, mid - 2 * sd
    df["bw"] = 4 * sd / mid
    df["hh20"], df["ll20"] = df.high.shift().rolling(20).max(), df.low.shift().rolling(20).min()
    df["hh55"], df["ll55"] = df.high.shift().rolling(55).max(), df.low.shift().rolling(55).min()
    return df


# Cada estrategia devuelve (lado, distancia_stop) o None, mirando solo velas cerradas.
def ruptura_tendencia(r, p):
    """Rompe máximo/mínimo de 20 velas con vela grande, a favor de la EMA 200."""
    big = abs(r.close - r.open) > r.atr
    if big and r.close > r.ema200 and r.close > r.hh20:
        return 1, 2 * r.atr
    if big and r.close < r.ema200 and r.close < r.ll20:
        return -1, 2 * r.atr


def retroceso_en_tendencia(r, p):
    """Tendencia alcista/bajista clara y el RSI sale de sobreventa/sobrecompra."""
    if r.ema50 > r.ema200 and r.close > r.ema200 and p.rsi < 35 <= r.rsi:
        return 1, 2 * r.atr
    if r.ema50 < r.ema200 and r.close < r.ema200 and p.rsi > 65 >= r.rsi:
        return -1, 2 * r.atr


def compresion_expansion(r, p):
    """Volatilidad comprimida (bandas de Bollinger estrechas) y ruptura fuerte."""
    if p.bw > p.bw_q20:
        return None
    if r.close > r.bb_up and r.close > r.ema50:
        return 1, 2 * r.atr
    if r.close < r.bb_dn and r.close < r.ema50:
        return -1, 2 * r.atr


def tortuga_55(r, p):
    """Ruptura de 55 velas (tramos largos), estilo Turtle."""
    if r.close > r.hh55:
        return 1, 2 * r.atr
    if r.close < r.ll55:
        return -1, 2 * r.atr


ESTRATEGIAS = {
    "Ruptura con tendencia": ruptura_tendencia,
    "Retroceso en tendencia": retroceso_en_tendencia,
    "Compresión y expansión": compresion_expansion,
    "Ruptura 55 (tortuga)": tortuga_55,
}


def run(df, signal, start=None, end=None, trail=True):
    df = df.copy()
    df["bw_q20"] = df.bw.rolling(120).quantile(0.2)
    d = df.loc[start:end].dropna()
    eq, pos, trades, prev = 10_000.0, None, [], None
    for r in d.itertuples():
        if pos:
            side, entry, sl, tp, size, stop_d = pos
            if (side == 1 and r.low <= sl) or (side == -1 and r.high >= sl):
                px = sl                      # peor caso: el stop se toca primero
            elif (side == 1 and r.high >= tp) or (side == -1 and r.low <= tp):
                px = tp
            else:
                px = None
                if trail:                    # sale cuando el mercado se voltea
                    new = r.close - side * 3 * r.atr
                    sl = max(sl, new) if side == 1 else min(sl, new)
                    pos = (side, entry, sl, tp, size, stop_d)
            if px is not None:
                pnl = (px - entry) * side * size - SPREAD * size
                eq += pnl
                trades.append((r.Index, side, pnl, pnl / (size * stop_d)))
                pos = None
        if pos is None and prev is not None:
            s = signal(r, prev)
            if s:
                side, stop_d = s
                size = eq * RISK / stop_d
                pos = (side, r.close, r.close - side * stop_d, r.close + side * RR * stop_d, size, stop_d)
        prev = r
    return eq, pd.DataFrame(trades, columns=["fecha", "lado", "pnl", "R"])


def stats(eq, t):
    if t.empty:
        return dict(ops=0)
    curve = pd.concat([pd.Series([10_000.0]), 10_000 + t.pnl.cumsum()])
    dd = ((curve - curve.cummax()) / curve.cummax()).min()
    losses = -t[t.pnl <= 0].pnl.sum()
    return dict(ops=len(t), acierto=(t.pnl > 0).mean(), pf=t[t.pnl > 0].pnl.sum() / losses if losses else float("inf"),
                retorno=eq / 10_000 - 1, maxdd=dd)


def fmt(name, s):
    if not s["ops"]:
        return f"  {name:26s} sin operaciones"
    return (f"  {name:26s} ops={s['ops']:3d} acierto={s['acierto']:5.1%} PF={s['pf']:4.2f} "
            f"retorno={s['retorno']:7.1%} maxDD={s['maxdd']:6.1%}")


if __name__ == "__main__":
    old = load("data/XAUUSDh4.csv", scale=100)
    new = load("data/XAUUSD_4h_2025_2026.csv")
    start = new.index[-1] - pd.DateOffset(years=1)
    bh = new.loc[start:].close
    print("A) 2012-2022 (validación, 10.000 USD)")
    for n, f in ESTRATEGIAS.items():
        print(fmt(n, stats(*run(old, f))))
    print(f"\nB) Último año {start:%Y-%m-%d} a {new.index[-1]:%Y-%m-%d} (10.000 USD)")
    print(f"  {'Comprar y mantener oro':26s} retorno={bh.iloc[-1] / bh.iloc[0] - 1:7.1%}")
    for n, f in ESTRATEGIAS.items():
        print(fmt(n, stats(*run(new, f, start=start))))
