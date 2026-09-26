"""Backtest de la estrategia de oro del reel (XAUUSD, velas de 4 horas).

Reglas tomadas del video:
  - Sigue la tendencia en velas de 4H e ignora el ruido intradía.
  - Entra en un "movimiento grande" y sale cuando el mercado se voltea.
  - Stop loss del 1% en cada operación, sin excepciones.
  - Take profit con ratio 1:4.

El video no da indicadores concretos, así que se definen así (supuestos):
  - Tendencia: precio sobre/bajo la EMA 200.
  - Movimiento grande: cierre que rompe el máximo/mínimo de las últimas N velas
    con un cuerpo de vela mayor a 1 ATR.
  - Mercado se voltea: cierre que cruza la EMA 50 en contra.
  - Stop 1% del precio de entrada; TP a 4% (1:4). Tamaño para arriesgar 1% de la cuenta.
"""
import sys
import pandas as pd

DATA = "data/XAUUSDh4.csv"
SPREAD = 0.35          # USD por onza, costo de ida y vuelta
RISK = 0.01            # 1% de la cuenta por operación
SL_PCT = 0.01          # stop 1% del precio
RR = 4                 # take profit 1:4


def load():
    df = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")
    df[["open", "high", "low", "close"]] /= 100  # el CSV viene x100
    return df


def signals(df, n=20):
    c = df.close
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    big = (c - df.open).abs() > df.atr
    df["long"] = (c > df.ema200) & (c > df.high.shift().rolling(n).max()) & big
    df["short"] = (c < df.ema200) & (c < df.low.shift().rolling(n).min()) & big
    return df


def run(df, reversal_exit=True, start=None, end=None):
    d = df.loc[start:end]
    eq, pos, trades = 10_000.0, None, []
    rows = d.itertuples()
    for r in rows:
        if pos:
            side, entry, sl, tp, size = pos
            hit_sl = r.low <= sl if side == 1 else r.high >= sl
            hit_tp = r.high >= tp if side == 1 else r.low <= tp
            exit_px = None
            if hit_sl:                      # peor caso: stop primero
                exit_px = sl
            elif hit_tp:
                exit_px = tp
            elif reversal_exit and ((side == 1 and r.close < r.ema50) or (side == -1 and r.close > r.ema50)):
                exit_px = r.close
            if exit_px is not None:
                pnl = (exit_px - entry) * side * size - SPREAD * size
                eq += pnl
                trades.append((r.Index, side, pnl))
                pos = None
        if pos is None and (r.long or r.short):
            side = 1 if r.long else -1
            entry = r.close
            sl = entry * (1 - SL_PCT * side)
            tp = entry * (1 + SL_PCT * RR * side)
            size = eq * RISK / (entry * SL_PCT)
            pos = (side, entry, sl, tp, size)
    return eq, pd.DataFrame(trades, columns=["fecha", "lado", "pnl"])


def report(name, eq, t):
    if t.empty:
        print(f"{name}: sin operaciones"); return
    curve = 10_000 + t.pnl.cumsum()
    dd = ((curve - curve.cummax()) / curve.cummax()).min()
    wins = t[t.pnl > 0]
    pf = wins.pnl.sum() / -t[t.pnl <= 0].pnl.sum()
    years = (t.fecha.iloc[-1] - t.fecha.iloc[0]).days / 365.25
    cagr = (eq / 10_000) ** (1 / years) - 1 if years > 0 else 0
    print(f"{name:32s} ops={len(t):4d} acierto={len(wins)/len(t):5.1%} PF={pf:4.2f} "
          f"retorno={eq/10_000-1:7.1%} anual={cagr:6.1%} maxDD={dd:6.1%}")


if __name__ == "__main__":
    df = signals(load())
    print(f"Datos: {df.index[0]:%Y-%m-%d} a {df.index[-1]:%Y-%m-%d}, capital inicial 10.000 USD\n")
    for rev in (False, True):
        tag = "SL1%/TP4% + salida por giro" if rev else "SL1%/TP4% puro"
        eq, t = run(df, rev)
        report(tag, eq, t)
        if "-v" in sys.argv:
            for y, g in t.groupby(t.fecha.dt.year):
                print(f"   {y}: ops={len(g):3d} pnl={g.pnl.sum():9.0f}")
