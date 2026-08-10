#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIST ADAPTIVE ML v3.5.2 EXPLAIN — GitHub Actions paper automation.

Kilitli strateji çekirdeği korunur:
- Adaptif XGBoost
- 350 hisse
- 730 gün kayan eğitim penceresi
- 120 gün recency half-life
- 5 işlem gününde bir yeniden eğitim
- TP +%5 / SL -%5 / maksimum 5 işlem günü
- Top-3 gösterim, yalnız #1 PRIMARY paper sinyali

Scheduled PRIMARY: kapanış sonrası repo state'ini günceller.
Manual PREVIEW: gün içi dahil çalışabilir, hiçbir kalıcı paper/model state'i değiştirmez.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT/'state'
DATA_DIR = ROOT/'data'
OUTPUT_DIR = ROOT/'output'
for d in (STATE_DIR, DATA_DIR, OUTPUT_DIR): d.mkdir(parents=True, exist_ok=True)

PERIOD = '5y'
TRAIN_WINDOW_DAYS = 730
RECENCY_HALF_LIFE_DAYS = 120
RETRAIN_EVERY_TRADING_DAYS = 5
TP_PCT = 0.05
SL_PCT = 0.05
HOLD_DAYS = 5
COMMISSION_PER_SIDE = 0.0010
SLIPPAGE_PER_SIDE = 0.0005
UNIVERSE_SIZE = 350
LIQUIDITY_PERIOD = '18mo'
LIQUIDITY_DAYS = 60
UNIVERSE_MIN_RAW_ROWS = 300
UNIVERSE_MIN_ACTIVE_RATIO = 0.90
BATCH_SIZE = 70
MAX_DAILY_CANDIDATES = 3
COOLDOWN_TRADING_DAYS = 5
PAPER_INITIAL_CAPITAL = 100_000.0
PAPER_RISK_PER_TRADE = 0.01
PAPER_MAX_OPEN_POSITIONS = 5
PAPER_VERSION = 'v3.5.1'       # model/strategy signature — değiştirme
UI_VERSION = 'v3.5.2 EXPLAIN GITHUB'
TZ = ZoneInfo('Europe/Istanbul')

WATCHLIST_FILE = DATA_DIR/'daily_top3.csv'
PAPER_SIGNAL_FILE = DATA_DIR/'paper_signals.csv'
PAPER_TRADES_FILE = DATA_DIR/'paper_trades.csv'
PAPER_EQUITY_FILE = DATA_DIR/'paper_equity.csv'
PAPER_SUMMARY_FILE = DATA_DIR/'paper_summary.csv'
LIVE_MODEL_FILE = STATE_DIR/'adaptive_live_model.json'
LIVE_MODEL_META_FILE = STATE_DIR/'adaptive_live_model_meta.json'
REPORT_FILE = OUTPUT_DIR/'latest_report.md'

CORE_SYMBOLS = ['THYAO','GARAN','ASELS','SASA','EKGYO','SAHOL','KCHOL','TUPRS','PETKM','EREGL','TRALT','BIMAS','YKBNK','AKBNK','TRMET','TCELL','VESTL','SISE','HEKTS','OTKAR','TOASO','FROTO','ARCLK','AEFES','CCOLA','KONYA','TURSG']

MASTER_BIST_LIST = [
    'A1CAP','ACSEL','ADEL','ADESE','ADGYO','AEFES','AFYON','AGESA','AGHOL','AGYO',
    'AHGAZ','AKBNK','AKCNS','AKENR','AKFGY','AKFYE','AKGRT','AKMGY','AKSA','AKSEN',
    'AKSGY','AKSUE','AKYHO','ALARK','ALBRK','ALCAR','ALCTL','ALFAS','ALGYO','ALKA',
    'ALKIM','ALMAD','ANELE','ANGEN','ANHYT','ANSGR','ARASE','ARCLK','ARDYZ','ARENA',
    'ARSAN','ARZUM','ASELS','ASGYO','ASTOR','ASUZU','ATAGY','ATAKP','ATATP','ATEKS',
    'ATLAS','ATSYH','AVGYO','AVHOL','AVOD','AVTUR','AYCES','AYDEM','AYEN','AYES',
    'AYGAZ','AZTEK','BAGFS','BAKAB','BALAT','BANVT','BARMA','BASCM','BASGZ','BAYRK',
    'BERA','BEYAZ','BFREN','BIENY','BIGCH','BIMAS','BIOEN','BIZIM','BLCYT','BMSCH',
    'BMSTL','BNTAS','BOBET','BOSSA','BRISA','BRKO','BRKSN','BRKVY','BRLSM','BRMEN',
    'BRSAN','BRYAT','BSOKE','BTCIM','BUCIM','BURCE','BURVA','BVSAN','BYDNR','CANTE',
    'CASA','CCOLA','CELHA','CEMAS','CEMTS','CEOEM','CIMSA','CLEBI','CMBTN','CMENT',
    'CONSE','COSMO','CRDFA','CRFSA','CUSAN','CVKMD','CWENE','DAGHL','DAGI','DAPGM',
    'DARDL','DENGE','DERHL','DERIM','DESA','DESPC','DEVA','DGATE','DGGYO','DGNMO',
    'DIRIT','DITAS','DMRGD','DMSAS','DNISI','DOAS','DOBUR','DOGUB','DOHOL','DOKTA',
    'DURDO','DYOBY','DZGYO','EBEBK','ECILC','ECZYT','EDATA','EDIP','EGEEN','EGEPO',
    'EGGUB','EGPRO','EGSER','EKGYO','EKIZ','EKSUN','ELITE','EMKEL','EMNIS','ENERY',
    'ENJSA','ENKAI','ENSRI','EPLAS','ERBOS','ERCB','EREGL','ERSU','ESCAR','ESCOM',
    'ESEN','ETILR','ETYAT','EUHOL','EUKYO','EUPWR','EUREN','EUYO','EYGYO','FADE',
    'FLAP','FMIZP','FONET','FORMT','FORTE','FRIGO','FROTO','FZLGY','GARAN','GARFA',
    'GEDIK','GEDZA','GENIL','GENTS','GEREL','GESAN','GIPTA','GLBMD','GLCVY','GLRYH',
    'GLYHO','GMTAS','GOKNR','GOLTS','GOODY','GOZDE','GRNYO','GRSEL','GRTRK','GSDDE',
    'GSDHO','GUBRF','GWIND','GZNMI','HALKB','HATEK','HATSN','HDFGS','HEDEF','HEKTS',
    'HKTM','HLGYO','HTTBT','HUBVC','HUNER','HURGZ','ICBCT','ICUGS','IDEAS','IDGYO',
    'IEYHO','IHAAS','IHEVA','IHGZT','IHLAS','IHLGM','IHYAY','IMASM','INDES','INFO',
    'INGRM','INTEM','INVEO','INVES','TRENJ','ISATR','ISBIR','ISBTR','ISCTR','ISDMR',
    'ISFIN','ISGSY','ISGYO','ISKPL','ISKUR','ISMEN','ISSEN','ISYAT','ITTFH','IZENR',
    'IZFAS','IZINV','IZMDC','JANTS','KAPLM','KAREL','KARSN','KARTN','KARYE','KATMR',
    'KCAER','KCHOL','KENT','KERVN','KERVT','KFEIN','KGYO','KIMMR','KLGYO','KLKIM',
    'KLMSN','KLNMA','KLRHO','KLSER','KLSYN','KMPUR','KNFRT','KONKA','KONTR','KONYA',
    'KOPOL','KORDS','TRMET','TRALT','KRDMA','KRDMB','KRDMD','KRGYO','KRONT','KRPLS',
    'KRSTL','KRTEK','KRVGD','KSTUR','KTLEV','KTSKR','KUTPO','KUVVA','KUYAS','KZBGY',
    'KZGYO','LIDER','LIDFA','LINK','LKMNH','LOGO','LUKSK','MAALT','MACKO','MAGEN',
    'MAKIM','MAKTK','MANAS','MARKA','MAVI','MEDTR','MEGAP','MEPET','MERCN','MERIT',
    'METRO','METUR','MGROS','MIATK','MIPAZ','MMCAS','MNDRS','MNDTR','MOBTL','MPARK',
    'MRSHL','MSGYO','MTRKS','MTRYO','MZHLD','NATEN','NETAS','NIBAS','NTGAZ','NTHOL',
    'NUGYO','NUHCM','OBASE','ODAS','OFSYM','ONCSM','ORCAY','ORGE','ORMA','OSMEN',
    'OSTIM','OTKAR','OTTO','OYAKC','OYAYO','OYLUM','OYYAT','OZGYO','OZKGY','OZRDN',
    'OZSUB','PAGYO','PAMEL','PAPIL','PARSN','PASEU','PCILT','PEGYO','PEKGY','PENGD',
    'PENTA','PETKM','PETUN','PGSUS','PINSU','PKART','PKENT','PLTUR','PNLSN','PNSUT',
    'POLHO','POLTK','PRDGS','PRKAB','PRKME','PRZMA','PSDTC','PSGYO','QNBFB','QNBFL',
    'QUAGR','RALYH','RAYSG','REEDR','RNPOL','RODRG','RTALB','RUBNS','RYGYO','RYSAS',
    'SAFKR','SAHOL','SAMAT','SANEL','SANFM','SANKO','SARKY','SASA','SAYAS','SDTTR',
    'SEGYO','SEKFK','SEKUR','SELEC','SELGD','SELVA','SEYKM','SILVR','SISE','SKBNK',
    'SKTAS','SMART','SMRTG','SNGYO','SNICA','SNKRN','SNPAM','SODSN','SOKE','SOKM',
    'SONME','SRVGY','SUMAS','SUNTK','SUWEN','TARKM','TATEN','TATGD','TAVHL','TBORG',
    'TCELL','TDGYO','TEKTU','TERA','TETMT','TEZOL','TGSAS','THYAO','TKFEN','TKNSA',
    'TLMAN','TMPOL','TMSN','TNZTP','TOASO','TRCAS','TRGYO','TRILC','TSGYO','TSKB',
    'TTKOM','TTRAK','TUCLK','TUKAS','TUPRS','TUREX','TURGG','TURSG','UFUK','ULAS',
    'ULKER','ULUFA','ULUSE','ULUUN','UMPAS','UNLU','USAK','UZERB','VAKBN','VAKFN',
    'VAKKO','VANGD','VBTYZ','VERTU','VERUS','VESBE','VESTL','VKFYO','VKGYO','VKING',
    'YAPRK','YATAS','YAYLA','YBTAS','YEOTK','YESIL','YGGYO','YGYO','YKBNK','YKSLN',
    'YONGA','YUNSA','YYAPI','YYLGD','ZEDUR','ZOREN','ZRGYO'
]

YAHOO_SYMBOLS = {
    'TRALT': ['KOZAL.IS', 'TRALT.IS'],
    'TRMET': ['KOZAA.IS', 'TRMET.IS'],
    'TRENJ': ['IPEKE.IS', 'TRENJ.IS'],
}


def _yahoo_aliases(name):
    return YAHOO_SYMBOLS.get(name, [name + '.IS'])


def _clean_ohlcv(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    need = ['Open','High','Low','Close','Volume']
    if any(c not in out.columns for c in need):
        return pd.DataFrame()
    out = out[need].copy()
    for c in need:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(subset=['Close']).sort_index()
    try:
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
    except Exception:
        pass
    return out[~out.index.duplicated(keep='last')]


def _extract_ticker_frame(batch, ticker):
    """yf.download MultiIndex yönü değişse bile tek ticker OHLCV çıkarır."""
    if batch is None or batch.empty:
        return pd.DataFrame()
    cols = batch.columns
    if isinstance(cols, pd.MultiIndex):
        lv0 = cols.get_level_values(0)
        lv1 = cols.get_level_values(1)
        try:
            if ticker in lv0:
                return _clean_ohlcv(batch[ticker])
            if ticker in lv1:
                return _clean_ohlcv(batch.xs(ticker, axis=1, level=1))
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()
    return _clean_ohlcv(batch)


def download_many_history(names, period='5y', chunk_size=BATCH_SIZE):
    """
    Yüzlerce hisseyi tek tek istemek yerine Yahoo'dan batch indirir.
    Aliaslı hisselerde eski+yeni sembol parçalarını birleştirir.
    """
    names = list(dict.fromkeys(names))
    aliases = {name: _yahoo_aliases(name) for name in names}
    yahoo_tickers = list(dict.fromkeys(t for arr in aliases.values() for t in arr))
    ticker_frames = {}

    for i in range(0, len(yahoo_tickers), chunk_size):
        chunk = yahoo_tickers[i:i+chunk_size]
        try:
            batch = yf.download(
                tickers=chunk,
                period=period,
                interval='1d',
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=True,
                group_by='ticker',
            )
        except Exception as e:
            print(f'⚠️ Batch {i//chunk_size+1}: {type(e).__name__}')
            batch = pd.DataFrame()

        for ticker in chunk:
            fr = _extract_ticker_frame(batch, ticker)
            if not fr.empty:
                ticker_frames[ticker] = fr

    result = {}
    for name in names:
        parts = [ticker_frames[t] for t in aliases[name] if t in ticker_frames and not ticker_frames[t].empty]
        if not parts:
            result[name] = pd.DataFrame()
            continue
        out = pd.concat(parts).sort_index()
        result[name] = out[~out.index.duplicated(keep='last')]
    return result




# ==========================================================
# 3) FEATURE + GERÇEK İŞLEM HEDEFİ
# ==========================================================
BASE_FEATURES = [
    'RET_1','OPEN_GAP','HIGH_REL','LOW_REL','RSI',
    'MACD_PCT','MACD_SIGNAL_PCT','MACD_HIST_PCT',
    'BB_POS','SMA20_DIST','EMA20_DIST','EMA50_DIST','EMA20_SLOPE_3',
    'MOM_5','MOM_10','VOL_RATIO','ATR_PCT'
]
MARKET_FEATURES = [
    'MK_RET1','MK_MOM5','MK_MOM10','MK_BREADTH20',
    'MK_BREADTH50','MK_ATR','MK_DISP','MK_VOLRATIO'
]
FEATURES = BASE_FEATURES + MARKET_FEATURES

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return (100 - 100/(1 + ag/al.replace(0, np.nan))).fillna(50)

def _net_ret(entry_raw, exit_raw):
    entry_exec = entry_raw * (1 + SLIPPAGE_PER_SIDE)
    exit_exec = exit_raw * (1 - SLIPPAGE_PER_SIDE)
    entry_cost = entry_exec * (1 + COMMISSION_PER_SIDE)
    exit_value = exit_exec * (1 - COMMISSION_PER_SIDE)
    return exit_value / entry_cost - 1

def prepare_features(raw):
    g = raw.copy().sort_index()
    for c in ['Open','High','Low','Close','Volume']:
        g[c] = pd.to_numeric(g[c], errors='coerce')
    g = g.dropna(subset=['Open','High','Low','Close','Volume'])
    c = g['Close']; pc = c.shift(1)

    g['RSI'] = calculate_rsi(c)
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12-e26
    ms = macd.ewm(span=9, adjust=False).mean()
    mh = macd-ms
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    sm20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std(ddof=0)
    lo = sm20 - 2*sd20
    up = sm20 + 2*sd20
    bw = (up-lo).replace(0,np.nan)
    tr = pd.concat([
        g['High']-g['Low'],
        (g['High']-pc).abs(),
        (g['Low']-pc).abs()
    ],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()

    g['RET_1'] = c.pct_change()*100
    g['OPEN_GAP'] = (g['Open']/pc-1)*100
    g['HIGH_REL'] = (g['High']/c-1)*100
    g['LOW_REL'] = (g['Low']/c-1)*100
    g['MACD_PCT'] = macd/c*100
    g['MACD_SIGNAL_PCT'] = ms/c*100
    g['MACD_HIST_PCT'] = mh/c*100
    g['BB_POS'] = (c-lo)/bw
    g['SMA20_DIST'] = (c/sm20-1)*100
    g['EMA20_DIST'] = (c/e20-1)*100
    g['EMA50_DIST'] = (c/e50-1)*100
    g['EMA20_SLOPE_3'] = e20.pct_change(3)*100
    g['MOM_5'] = c.pct_change(5)*100
    g['MOM_10'] = c.pct_change(10)*100
    g['VOL_RATIO'] = g['Volume']/g['Volume'].rolling(5).mean()
    g['ATR_PCT'] = atr/c*100
    g[BASE_FEATURES] = g[BASE_FEATURES].replace([np.inf,-np.inf],np.nan)
    g = g.dropna(subset=BASE_FEATURES).copy()

    g['Target'] = np.nan
    g['TradeRet'] = np.nan
    g['ExitDate'] = pd.NaT
    g['ExitReason'] = None

    O,H,L,C = [g[x].to_numpy(float) for x in ['Open','High','Low','Close']]
    idx = g.index.to_numpy()
    for i in range(len(g)-HOLD_DAYS):
        entry = O[i+1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        tp = entry*(1+TP_PCT)
        sl = entry*(1-SL_PCT)
        exit_raw = C[i+HOLD_DAYS]
        exit_i = i+HOLD_DAYS
        reason = 'TIME'
        for j in range(i+1, i+1+HOLD_DAYS):
            o,h,l = O[j],H[j],L[j]
            if o <= sl:
                exit_raw, exit_i, reason = o, j, 'SL_GAP'; break
            if o >= tp:
                exit_raw, exit_i, reason = o, j, 'TP_GAP'; break
            if l <= sl and h >= tp:
                exit_raw, exit_i, reason = sl, j, 'SL_SAME_BAR'; break
            if l <= sl:
                exit_raw, exit_i, reason = sl, j, 'SL'; break
            if h >= tp:
                exit_raw, exit_i, reason = tp, j, 'TP'; break
        net = _net_ret(entry, exit_raw)
        g.iloc[i, g.columns.get_loc('TradeRet')] = net
        g.iloc[i, g.columns.get_loc('Target')] = float(net > 0)
        g.iloc[i, g.columns.get_loc('ExitDate')] = idx[exit_i]
        g.iloc[i, g.columns.get_loc('ExitReason')] = reason

    return g




def atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def select_universe():
    print('='*100)
    print(f'🌐 GENİŞ BIST EVRENİ TARANIYOR — {len(MASTER_BIST_LIST)} aday')
    print('='*100)
    recent_cache = download_many_history(MASTER_BIST_LIST, period=LIQUIDITY_PERIOD)
    rank_rows = []
    for symbol in MASTER_BIST_LIST:
        raw = recent_cache.get(symbol, pd.DataFrame())
        if raw.empty: continue
        valid = raw[['Close','Volume']].dropna().copy()
        valid = valid[(valid['Close'] > 0) & (valid['Volume'] >= 0)]
        if len(valid) < 60: continue
        recent = valid.tail(LIQUIDITY_DAYS)
        active_ratio = float((recent['Volume'] > 0).mean())
        turnover = (recent['Close'] * recent['Volume']).replace([np.inf,-np.inf], np.nan).dropna()
        if turnover.empty: continue
        rank_rows.append({
            'Hisse':symbol,'RawRows':int(len(valid)),'AktifOran':active_ratio,
            'MedyanHacimTL':float(turnover.median()),'OrtHacimTL':float(turnover.mean()),
            'SonFiyat':float(recent['Close'].dropna().iloc[-1]),
            'YeterliTarih':bool(len(valid)>=UNIVERSE_MIN_RAW_ROWS),
        })
    ranking = pd.DataFrame(rank_rows)
    if ranking.empty: raise RuntimeError('Likidite evreni oluşturulamadı.')
    ranking = ranking.sort_values(['YeterliTarih','AktifOran','MedyanHacimTL','OrtHacimTL'], ascending=[False,False,False,False]).reset_index(drop=True)
    eligible = ranking[(ranking['YeterliTarih']) & (ranking['AktifOran']>=UNIVERSE_MIN_ACTIVE_RATIO)].sort_values('MedyanHacimTL',ascending=False)
    selected = eligible['Hisse'].head(UNIVERSE_SIZE).tolist()
    if len(selected)<UNIVERSE_SIZE:
        selected += [s for s in ranking['Hisse'].tolist() if s not in selected][:UNIVERSE_SIZE-len(selected)]
    if len(selected)<UNIVERSE_SIZE:
        selected += [s for s in MASTER_BIST_LIST if s not in selected][:UNIVERSE_SIZE-len(selected)]
    for core in CORE_SYMBOLS:
        if core in MASTER_BIST_LIST and core not in selected:
            for j in range(len(selected)-1,-1,-1):
                if selected[j] not in CORE_SYMBOLS:
                    selected[j]=core; break
    selected = list(dict.fromkeys(selected))
    if len(selected)<UNIVERSE_SIZE:
        selected += [s for s in ranking['Hisse'].tolist() if s not in selected][:UNIVERSE_SIZE-len(selected)]
    out = selected[:UNIVERSE_SIZE]
    print(f'✅ Seçilen hisse: {len(out)} | çekirdek: {sum(s in out for s in CORE_SYMBOLS)}/{len(CORE_SYMBOLS)}')
    return out, ranking


def build_all_data(bist_list):
    raw_history_cache = download_many_history(bist_list, period=PERIOD)
    data_cache = {}
    for symbol in bist_list:
        raw = raw_history_cache.get(symbol, pd.DataFrame())
        if raw.empty: continue
        d = prepare_features(raw)
        if len(d)>=260: data_cache[symbol]=d
    frames=[]
    for symbol,d in data_cache.items():
        t=d.reset_index()
        first=t.columns[0]
        if first!='Tarih': t=t.rename(columns={first:'Tarih'})
        t['Hisse']=symbol
        frames.append(t)
    if not frames: raise RuntimeError('Feature verisi üretilemedi.')
    all_data=pd.concat(frames,ignore_index=True)
    market=all_data.groupby('Tarih').agg(
        MK_RET1=('RET_1','median'), MK_MOM5=('MOM_5','median'), MK_MOM10=('MOM_10','median'),
        MK_BREADTH20=('EMA20_DIST',lambda x:(x>0).mean()), MK_BREADTH50=('EMA50_DIST',lambda x:(x>0).mean()),
        MK_ATR=('ATR_PCT','median'), MK_DISP=('RET_1','std'), MK_VOLRATIO=('VOL_RATIO','median'),
    ).reset_index()
    all_data=all_data.merge(market,on='Tarih',how='left')
    all_data[FEATURES]=all_data[FEATURES].replace([np.inf,-np.inf],np.nan)
    all_data=all_data.dropna(subset=FEATURES).sort_values(['Tarih','Hisse']).reset_index(drop=True)
    all_data['Tarih']=pd.to_datetime(all_data['Tarih']).dt.tz_localize(None)
    print(f'✅ Hazır veri: {len(all_data):,} satır | {all_data.Hisse.nunique()} hisse | {all_data.Tarih.min().date()} → {all_data.Tarih.max().date()}')
    return all_data, raw_history_cache


def build_model(y):
    neg=int((y==0).sum()); pos=int((y==1).sum())
    return xgb.XGBClassifier(
        n_estimators=450,max_depth=3,learning_rate=0.025,subsample=0.85,colsample_bytree=0.85,
        min_child_weight=5,reg_alpha=0.15,reg_lambda=1.75,objective='binary:logistic',
        eval_metric='logloss',random_state=SEED,n_jobs=-1,tree_method='hist',scale_pos_weight=neg/max(1,pos)
    )


def adaptive_fit(all_data, train_end):
    train_end=pd.Timestamp(train_end)
    tr=all_data[all_data['Target'].notna() & (all_data['ExitDate']<=train_end) & (all_data['Tarih']>=train_end-pd.Timedelta(days=TRAIN_WINDOW_DAYS))].copy()
    if tr.empty: raise RuntimeError('Adaptif eğitim satırı yok.')
    X=tr[FEATURES].to_numpy(np.float32); y=tr['Target'].to_numpy(np.float32)
    age=(train_end-tr['Tarih']).dt.days.clip(lower=0).to_numpy(float)
    sw=np.exp(-np.log(2)*age/RECENCY_HALF_LIFE_DAYS)
    model=build_model(y); model.fit(X,y,sample_weight=sw)
    return model,len(tr)


def feature_signature():
    payload=json.dumps({'features':FEATURES,'train_window_days':TRAIN_WINDOW_DAYS,'half_life_days':RECENCY_HALF_LIFE_DAYS,
        'retrain_every':RETRAIN_EVERY_TRADING_DAYS,'tp':TP_PCT,'sl':SL_PCT,'hold':HOLD_DAYS,'seed':SEED,'version':PAPER_VERSION},sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def previous_market_day(day, market_days):
    day=pd.Timestamp(day).normalize(); prev=market_days[market_days<day]
    return pd.Timestamp(prev[-1]).normalize() if len(prev) else day


def market_day_distance(start_day,end_day,market_days):
    start_day=pd.Timestamp(start_day).normalize(); end_day=pd.Timestamp(end_day).normalize()
    idx={pd.Timestamp(d).normalize():i for i,d in enumerate(market_days)}
    if start_day not in idx or end_day not in idx: return None
    return idx[end_day]-idx[start_day]


def load_meta():
    if not LIVE_MODEL_META_FILE.exists(): return None
    try: return json.loads(LIVE_MODEL_META_FILE.read_text(encoding='utf-8'))
    except Exception: return None


def get_live_model(all_data, latest_date, market_days, persist: bool):
    meta=load_meta(); need=False; reason=''
    if not LIVE_MODEL_FILE.exists() or meta is None: need,reason=True,'MODEL_OR_META_MISSING'
    elif meta.get('strategy_version')!=PAPER_VERSION: need,reason=True,'VERSION_CHANGED'
    elif meta.get('feature_signature')!=feature_signature(): need,reason=True,'FEATURE_OR_PARAM_CHANGED'
    else:
        dist=market_day_distance(meta.get('block_start_date'),latest_date,market_days)
        if dist is None or dist<0: need,reason=True,'TRADING_CALENDAR_MISMATCH'
        elif dist>=RETRAIN_EVERY_TRADING_DAYS: need,reason=True,f'{dist}_TRADING_DAYS_SINCE_BLOCK_START'
    if need:
        train_end=previous_market_day(latest_date,market_days)
        model,nrows=adaptive_fit(all_data,train_end)
        meta={'strategy_version':PAPER_VERSION,'feature_signature':feature_signature(),'block_start_date':pd.Timestamp(latest_date).strftime('%Y-%m-%d'),
              'trained_through':pd.Timestamp(train_end).strftime('%Y-%m-%d'),'train_rows':int(nrows),'retrain_every_trading_days':RETRAIN_EVERY_TRADING_DAYS,
              'saved_at':datetime.now(TZ).isoformat(),'reason':reason}
        if persist:
            tmp=LIVE_MODEL_FILE.with_suffix('.json.tmp')
            model.save_model(str(tmp)); os.replace(tmp,LIVE_MODEL_FILE)
            atomic_write_text(LIVE_MODEL_META_FILE,json.dumps(meta,ensure_ascii=False,indent=2))
        return model,meta,True
    model=xgb.XGBClassifier(); model.load_model(str(LIVE_MODEL_FILE))
    return model,meta,False

FEATURE_LABELS = {
    'RET_1':'1 günlük getiri',
    'OPEN_GAP':'Açılış gap',
    'HIGH_REL':'Gün içi tepe/kapanış farkı',
    'LOW_REL':'Gün içi dip/kapanış farkı',
    'RSI':'RSI(14)',
    'MACD_PCT':'MACD',
    'MACD_SIGNAL_PCT':'MACD sinyal',
    'MACD_HIST_PCT':'MACD histogram',
    'BB_POS':'Bollinger bant konumu',
    'SMA20_DIST':'SMA20 mesafesi',
    'EMA20_DIST':'EMA20 mesafesi',
    'EMA50_DIST':'EMA50 mesafesi',
    'EMA20_SLOPE_3':'EMA20 3 günlük eğim',
    'MOM_5':'5 günlük momentum',
    'MOM_10':'10 günlük momentum',
    'VOL_RATIO':'Hacim oranı',
    'ATR_PCT':'ATR(14) volatilite',
    'MK_RET1':'Piyasa medyan 1g getiri',
    'MK_MOM5':'Piyasa medyan 5g momentum',
    'MK_MOM10':'Piyasa medyan 10g momentum',
    'MK_BREADTH20':'Piyasa EMA20 breadth',
    'MK_BREADTH50':'Piyasa EMA50 breadth',
    'MK_ATR':'Piyasa medyan ATR',
    'MK_DISP':'Piyasa getiri dağılımı',
    'MK_VOLRATIO':'Piyasa medyan hacim oranı',
}

_PERCENT_FEATURES = {
    'RET_1','OPEN_GAP','HIGH_REL','LOW_REL','MACD_PCT','MACD_SIGNAL_PCT',
    'MACD_HIST_PCT','SMA20_DIST','EMA20_DIST','EMA50_DIST','EMA20_SLOPE_3',
    'MOM_5','MOM_10','ATR_PCT','MK_RET1','MK_MOM5','MK_MOM10','MK_ATR','MK_DISP'
}
_BREADTH_FEATURES = {'MK_BREADTH20','MK_BREADTH50'}

def _fmt_feature_value(feature, value):
    value = float(value)
    if feature in _BREADTH_FEATURES:
        return f'%{value*100:.1f}'
    if feature in _PERCENT_FEATURES:
        return f'%{value:.2f}'
    if feature == 'RSI':
        return f'{value:.1f}'
    if feature in {'VOL_RATIO','MK_VOLRATIO'}:
        return f'{value:.2f}x'
    if feature == 'BB_POS':
        return f'{value:.2f}'
    return f'{value:.3f}'

def _xgb_local_contributions(row):
    """
    XGBoost TreeSHAP katkıları.
    Pozitif katkı skoru yukarı, negatif katkı skoru aşağı iter.
    Bu açıklama nedensellik değil, model içi karar açıklamasıdır.
    """
    x = row[FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)
    booster = live_model.get_booster()
    feature_names = booster.feature_names
    dm = xgb.DMatrix(x, feature_names=feature_names if feature_names else None)
    vals = booster.predict(dm, pred_contribs=True)[0]
    effects = vals[:-1]  # son eleman bias/base value
    items = []
    for i, f in enumerate(FEATURES):
        items.append({
            'feature': f,
            'label': FEATURE_LABELS.get(f, f),
            'value': float(row[f]),
            'effect': float(effects[i]),
        })
    positive = sorted([z for z in items if z['effect'] > 0], key=lambda z:z['effect'], reverse=True)
    negative = sorted([z for z in items if z['effect'] < 0], key=lambda z:z['effect'])
    return positive, negative

def _reason_text(row, n=3):
    pos, _ = _xgb_local_contributions(row)
    if not pos:
        return 'Belirgin pozitif katkı ayrışmadı'
    return ' | '.join(
        f'{z["label"]}: {_fmt_feature_value(z["feature"], z["value"])}'
        for z in pos[:n]
    )




def load_primary_signals():
    if not PAPER_SIGNAL_FILE.exists(): return pd.DataFrame()
    ph=pd.read_csv(PAPER_SIGNAL_FILE)
    if ph.empty or not {'SignalDate','Hisse'}.issubset(ph.columns): return pd.DataFrame()
    ph['SignalDate']=pd.to_datetime(ph['SignalDate'],errors='coerce').dt.normalize()
    if 'CreatedAt' in ph.columns:
        ph['CreatedAt']=pd.to_datetime(ph['CreatedAt'],errors='coerce')
        ph=ph.sort_values(['SignalDate','CreatedAt'],ascending=[True,True],na_position='last')
    else: ph=ph.sort_values('SignalDate')
    return ph.dropna(subset=['SignalDate','Hisse']).drop_duplicates('SignalDate',keep='first').reset_index(drop=True)


def qualified_top3(today, primary_history, latest_date, market_days):
    prev=market_days[market_days<latest_date]
    cooldown_days=set(pd.Timestamp(x).normalize() for x in prev[-COOLDOWN_TRADING_DAYS:])
    recent=set()
    if len(primary_history):
        recent=set(primary_history.loc[primary_history['SignalDate'].isin(cooldown_days),'Hisse'].astype(str))
    q=today[~today['Hisse'].astype(str).isin(recent)].sort_values('AdaptiveScore',ascending=False).head(MAX_DAILY_CANDIDATES).copy()
    q['Rank']=np.arange(1,len(q)+1); q['RefTP']=q['Close']*(1+TP_PCT); q['RefSL']=q['Close']*(1-SL_PCT)
    q['AIWhy']=q.apply(lambda r:_reason_text(r,3),axis=1)
    return q


def update_watchlist(qualified, latest_date):
    if qualified.empty: return
    watch=qualified[['Rank','Hisse','Close','AdaptiveScore','RefTP','RefSL','AIWhy']].copy()
    watch.insert(0,'Tarih',latest_date.date()); watch['KayitZamani']=datetime.now(TZ).isoformat()
    if WATCHLIST_FILE.exists():
        old=pd.read_csv(WATCHLIST_FILE); watch=pd.concat([old,watch],ignore_index=True)
    watch['Tarih']=pd.to_datetime(watch['Tarih'],errors='coerce').dt.date
    watch=watch.drop_duplicates(['Tarih','Rank'],keep='last').sort_values(['Tarih','Rank'])
    atomic_write_csv(watch,WATCHLIST_FILE)


def add_primary_signal(primary_history, best, latest_date):
    new=pd.DataFrame([{'SignalDate':latest_date,'Hisse':str(best['Hisse']),'SignalClose':float(best['Close']),
        'AdaptiveScore':float(best['AdaptiveScore']),'ReferenceTP':float(best['RefTP']),'ReferenceSL':float(best['RefSL']),
        'AIWhy':str(best['AIWhy']),'CreatedAt':datetime.now(TZ).isoformat(),'StrategyVersion':PAPER_VERSION,'UIVersion':UI_VERSION}])
    ph=pd.concat([primary_history,new],ignore_index=True)
    ph['SignalDate']=pd.to_datetime(ph['SignalDate']).dt.normalize(); ph['CreatedAt']=pd.to_datetime(ph['CreatedAt'],errors='coerce')
    ph=ph.sort_values(['SignalDate','CreatedAt'],na_position='last').drop_duplicates('SignalDate',keep='first')
    atomic_write_csv(ph,PAPER_SIGNAL_FILE); return ph

# ==========================================================
# 8) GERÇEKÇİ PAPER TRADING MOTORU
# Her çalıştırmada paper_signals.csv'den hesabı baştan deterministik kurar.
# Böylece yarım/kırık state dosyası hesabı bozmaz.
# ==========================================================

# Drive ve dosya yolları HÜCRE 1'de zorunlu olarak tanımlandı.

def _paper_frame(symbol, cache):
    fr = cache.get(symbol, pd.DataFrame())
    if fr is None or fr.empty:
        return pd.DataFrame()
    fr = _clean_ohlcv(fr)
    if fr.empty:
        return fr
    fr = fr.copy()
    fr.index = pd.to_datetime(fr.index).tz_localize(None) if getattr(fr.index, 'tz', None) is not None else pd.to_datetime(fr.index)
    return fr.sort_index()


def _exit_exec_values(exit_raw, qty):
    exit_exec = float(exit_raw) * (1 - SLIPPAGE_PER_SIDE)
    proceeds = qty * exit_exec * (1 - COMMISSION_PER_SIDE)
    return exit_exec, proceeds


def _mark_liquidation_value(raw_close, qty):
    # Açık pozisyonu gün sonu equity'de bugünkü kapanıştan, tahmini çıkış maliyeti düşülmüş değerle işaretle.
    exit_exec = float(raw_close) * (1 - SLIPPAGE_PER_SIDE)
    return qty * exit_exec * (1 - COMMISSION_PER_SIDE)


def simulate_paper_account(signals, price_cache, asof_date):
    asof_date = pd.Timestamp(asof_date).normalize()
    sig = signals.copy()
    if sig.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sig['SignalDate'] = pd.to_datetime(sig['SignalDate'], errors='coerce').dt.normalize()
    sig['AdaptiveScore'] = pd.to_numeric(sig['AdaptiveScore'], errors='coerce')
    if 'CreatedAt' in sig.columns:
        sig['CreatedAt'] = pd.to_datetime(sig['CreatedAt'], errors='coerce')
        sig = sig.dropna(subset=['SignalDate','Hisse']).sort_values(['SignalDate','CreatedAt'], ascending=[True,True], na_position='last')
    else:
        sig = sig.dropna(subset=['SignalDate','Hisse']).sort_values(['SignalDate','AdaptiveScore'], ascending=[True,False])
    # Bir piyasa günü = tek PRIMARY. Eski hatalı çoklu kayıt varsa ilk oluşturulan korunur.
    sig = sig.drop_duplicates(subset=['SignalDate'], keep='first').reset_index(drop=True)
    sig['TradeID'] = [f'P{d.strftime("%Y%m%d")}_{s}' for d,s in zip(sig['SignalDate'],sig['Hisse'])]

    # Eski paper sinyallerinden biri güncel 350 evreninden düşmüşse onun verisini de indir.
    missing = [s for s in sig['Hisse'].astype(str).unique() if s not in price_cache or price_cache.get(s, pd.DataFrame()).empty]
    if missing:
        print(f'🔄 Eski paper pozisyonları için {len(missing)} eksik sembol verisi indiriliyor...')
        extra = download_many_history(missing, period=PERIOD)
        price_cache.update(extra)

    # Her sinyalin ilk gerçek sonraki işlem gününü bul.
    due_map = {}
    pending_no_bar = set()
    for _,r in sig.iterrows():
        symbol = str(r['Hisse'])
        fr = _paper_frame(symbol, price_cache)
        if fr.empty:
            pending_no_bar.add(r['TradeID'])
            continue
        future = fr.index[(fr.index > r['SignalDate']) & (fr.index <= asof_date)]
        if len(future):
            due_map.setdefault(pd.Timestamp(future[0]).normalize(), []).append(r.to_dict())
        else:
            pending_no_bar.add(r['TradeID'])

    # Simülasyon takvimi yalnız paper sinyali olan sembollerin gerçek işlem günlerinden oluşur.
    start = sig['SignalDate'].min()
    calendar = set()
    last_close = {}
    symbols = list(sig['Hisse'].astype(str).unique())
    for symbol in symbols:
        fr = _paper_frame(symbol, price_cache)
        if fr.empty:
            continue
        for d in fr.index[(fr.index >= start) & (fr.index <= asof_date)]:
            calendar.add(pd.Timestamp(d).normalize())
    calendar = sorted(calendar)

    cash = float(PAPER_INITIAL_CAPITAL)
    open_pos = {}
    records = {}
    equity_rows = []

    # Başlangıçta tüm sinyalleri pending olarak kayda al.
    for _,r in sig.iterrows():
        records[r['TradeID']] = {
            'TradeID': r['TradeID'], 'SignalDate': r['SignalDate'], 'Hisse': str(r['Hisse']),
            'AdaptiveScore': float(r['AdaptiveScore']) if pd.notna(r['AdaptiveScore']) else np.nan,
            'SignalClose': float(r.get('SignalClose', np.nan)) if pd.notna(r.get('SignalClose', np.nan)) else np.nan,
            'Status': 'PENDING', 'EntryDate': pd.NaT, 'EntryRaw': np.nan, 'EntryExec': np.nan,
            'Qty': 0, 'EntryCost': 0.0, 'TPRaw': np.nan, 'SLRaw': np.nan,
            'BarsHeld': 0, 'ExitDate': pd.NaT, 'ExitRaw': np.nan, 'ExitExec': np.nan,
            'ExitReason': '', 'ExitProceeds': 0.0, 'RealizedPnL': 0.0, 'NetReturn': np.nan,
        }

    def current_equity(mark_date, use_open=False):
        eq = cash
        for p in open_pos.values():
            fr = _paper_frame(p['Hisse'], price_cache)
            if fr.empty:
                continue
            sub = fr.loc[fr.index <= mark_date]
            if sub.empty:
                continue
            row = sub.iloc[-1]
            px = row['Open'] if use_open and pd.Timestamp(sub.index[-1]).normalize() == mark_date else row['Close']
            eq += _mark_liquidation_value(px, p['Qty'])
        return float(eq)

    for day in calendar:
        # 1) Önceden açık pozisyonlarda AÇILIŞ GAP çıkışlarını önce işle.
        for tid in list(open_pos.keys()):
            p = open_pos.get(tid)
            fr = _paper_frame(p['Hisse'], price_cache)
            if day not in fr.index:
                continue
            row = fr.loc[day]
            o = float(row['Open'])
            reason = None
            exit_raw = None
            if o <= p['SLRaw']:
                reason, exit_raw = 'SL_GAP', o
            elif o >= p['TPRaw']:
                reason, exit_raw = 'TP_GAP', o
            if reason:
                exit_exec, proceeds = _exit_exec_values(exit_raw, p['Qty'])
                cash += proceeds
                rec = records[tid]
                rec.update({
                    'Status':'CLOSED','ExitDate':day,'ExitRaw':exit_raw,'ExitExec':exit_exec,
                    'ExitReason':reason,'ExitProceeds':proceeds,
                    'RealizedPnL':proceeds-rec['EntryCost'],
                    'NetReturn':proceeds/rec['EntryCost']-1 if rec['EntryCost']>0 else np.nan,
                })
                del open_pos[tid]

        # 2) Bugün girişi gelen yeni paper sinyallerini açılıştan gir.
        for s in due_map.get(day, []):
            tid = s['TradeID']
            if records[tid]['Status'] != 'PENDING':
                continue
            if len(open_pos) >= PAPER_MAX_OPEN_POSITIONS:
                records[tid]['Status'] = 'SKIPPED_MAX_POS'
                records[tid]['ExitReason'] = 'MAX_OPEN_POSITIONS'
                continue
            symbol = str(s['Hisse'])
            fr = _paper_frame(symbol, price_cache)
            if fr.empty or day not in fr.index:
                continue
            entry_raw = float(fr.loc[day,'Open'])
            entry_exec = entry_raw * (1 + SLIPPAGE_PER_SIDE)
            unit_cost = entry_exec * (1 + COMMISSION_PER_SIDE)

            eq_open = current_equity(day, use_open=True)
            risk_budget = max(0.0, eq_open * PAPER_RISK_PER_TRADE)

            # Planlı stopta komisyon + slippage sonrası gerçek hisse-başı kaybı.
            # Böylece normal SL gerçekleşmesinde risk bütçesi yaklaşık %1'e daha doğru oturur.
            stop_raw = entry_raw * (1 - SL_PCT)
            stop_exec = stop_raw * (1 - SLIPPAGE_PER_SIDE)
            stop_value_per_share = stop_exec * (1 - COMMISSION_PER_SIDE)
            risk_per_share = max(unit_cost - stop_value_per_share, 0.0)
            qty_by_risk = int(np.floor(risk_budget / risk_per_share)) if risk_per_share > 0 else 0
            qty_by_cash = int(np.floor(cash / unit_cost)) if unit_cost > 0 else 0
            qty = min(qty_by_risk, qty_by_cash)
            if qty < 1:
                records[tid]['Status'] = 'SKIPPED_CASH'
                records[tid]['ExitReason'] = 'INSUFFICIENT_CASH'
                continue

            entry_cost = qty * unit_cost
            cash -= entry_cost
            p = {
                'TradeID':tid,'Hisse':symbol,'Qty':qty,'EntryDate':day,'EntryRaw':entry_raw,
                'EntryExec':entry_exec,'EntryCost':entry_cost,
                'TPRaw':entry_raw*(1+TP_PCT),'SLRaw':entry_raw*(1-SL_PCT),'BarsHeld':0,
            }
            open_pos[tid] = p
            records[tid].update({
                'Status':'OPEN','EntryDate':day,'EntryRaw':entry_raw,'EntryExec':entry_exec,
                'Qty':qty,'EntryCost':entry_cost,'TPRaw':p['TPRaw'],'SLRaw':p['SLRaw'],'BarsHeld':0,
            })

        # 3) Gün içi TP/SL ve 5. işlem günü zaman çıkışı.
        for tid in list(open_pos.keys()):
            p = open_pos.get(tid)
            fr = _paper_frame(p['Hisse'], price_cache)
            if day not in fr.index:
                continue
            row = fr.loc[day]
            h,l,c = float(row['High']),float(row['Low']),float(row['Close'])

            # Entry günü dahil, sembolün işlem gördüğü barları say.
            p['BarsHeld'] += 1
            records[tid]['BarsHeld'] = p['BarsHeld']

            reason = None
            exit_raw = None
            if l <= p['SLRaw'] and h >= p['TPRaw']:
                reason, exit_raw = 'SL_SAME_BAR', p['SLRaw']
            elif l <= p['SLRaw']:
                reason, exit_raw = 'SL', p['SLRaw']
            elif h >= p['TPRaw']:
                reason, exit_raw = 'TP', p['TPRaw']
            elif p['BarsHeld'] >= HOLD_DAYS:
                reason, exit_raw = 'TIME', c

            if reason:
                exit_exec, proceeds = _exit_exec_values(exit_raw, p['Qty'])
                cash += proceeds
                rec = records[tid]
                rec.update({
                    'Status':'CLOSED','ExitDate':day,'ExitRaw':exit_raw,'ExitExec':exit_exec,
                    'ExitReason':reason,'ExitProceeds':proceeds,
                    'RealizedPnL':proceeds-rec['EntryCost'],
                    'NetReturn':proceeds/rec['EntryCost']-1 if rec['EntryCost']>0 else np.nan,
                })
                del open_pos[tid]

        # 4) Gün sonu equity — açık pozisyonlar maliyetli likidasyon değeriyle işaretlenir.
        eq = current_equity(day, use_open=False)
        equity_rows.append({
            'Tarih':day,'Cash':cash,'Equity':eq,'OpenPositions':len(open_pos),
            'TotalReturn':eq/PAPER_INITIAL_CAPITAL-1,
        })

    trades = pd.DataFrame(list(records.values()))
    if not trades.empty:
        trades = trades.sort_values(['SignalDate','Hisse']).reset_index(drop=True)
    equity = pd.DataFrame(equity_rows)

    # Özet.
    closed = trades[trades['Status']=='CLOSED'].copy() if not trades.empty else pd.DataFrame()
    final_equity = float(equity['Equity'].iloc[-1]) if len(equity) else PAPER_INITIAL_CAPITAL
    if len(equity):
        peak = equity['Equity'].cummax()
        dd = equity['Equity']/peak-1
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    if len(closed):
        wins = closed.loc[closed['RealizedPnL']>0,'RealizedPnL'].sum()
        losses = -closed.loc[closed['RealizedPnL']<0,'RealizedPnL'].sum()
        pf = float(wins/losses) if losses>0 else np.inf
        wr = float((closed['RealizedPnL']>0).mean())
        avg_ret = float(closed['NetReturn'].mean())
        med_ret = float(closed['NetReturn'].median())
    else:
        pf,wr,avg_ret,med_ret = np.nan,np.nan,np.nan,np.nan

    summary = pd.DataFrame([{
        'AsOf':asof_date,'InitialCapital':PAPER_INITIAL_CAPITAL,'FinalEquity':final_equity,
        'TotalReturn':final_equity/PAPER_INITIAL_CAPITAL-1,'MaxDD':max_dd,
        'ClosedTrades':int(len(closed)),'OpenTrades':int((trades['Status']=='OPEN').sum()) if len(trades) else 0,
        'PendingTrades':int((trades['Status']=='PENDING').sum()) if len(trades) else 0,
        'SkippedTrades':int(trades['Status'].astype(str).str.startswith('SKIPPED').sum()) if len(trades) else 0,
        'WR':wr,'PF':pf,'AvgNetReturn':avg_ret,'MedianNetReturn':med_ret,
        'RiskPerTrade':PAPER_RISK_PER_TRADE,'MaxOpenPositions':PAPER_MAX_OPEN_POSITIONS,
    }])
    return trades,equity,summary





def persist_paper(primary_history, raw_history_cache, latest_date):
    if primary_history.empty:
        print('🧪 Paper trading: henüz birincil sinyal yok.'); return None,None,None
    trades,equity,summary=simulate_paper_account(primary_history,raw_history_cache,latest_date)
    atomic_write_csv(trades,PAPER_TRADES_FILE); atomic_write_csv(equity,PAPER_EQUITY_FILE); atomic_write_csv(summary,PAPER_SUMMARY_FILE)
    return trades,equity,summary


def make_report(mode, latest_date, model_retrained, meta, qualified, existing_today, trades=None, summary=None, stale=False):
    lines=[f'# BIST ADAPTIVE ML — {UI_VERSION}', '', f'- Çalışma modu: **{mode.upper()}**', f'- Son piyasa verisi: **{latest_date.date()}**',
           f'- Model: **{"yeniden eğitildi" if model_retrained else "kayıtlı 5-günlük blok modeli"}**', f'- Model blok başlangıcı: **{meta.get("block_start_date")}**',
           f'- TP/SL/Hold: **+%5 / -%5 / max 5 işlem günü**', '']
    if stale: lines += ['> ⚠️ PRIMARY için bugünün tamamlanmış günlük barı bulunamadı; yeni sinyal yazılmadı.','']
    if len(qualified):
        lines += ['## Top-3','', '|#|Hisse|ML skor|Sinyal fiyatı|Referans TP|Referans SL|AI neden?|','|---:|---|---:|---:|---:|---:|---|']
        for _,r in qualified.iterrows():
            why=str(r['AIWhy']).replace('|','/'); lines.append(f'|{int(r.Rank)}|{r.Hisse}|%{r.AdaptiveScore*100:.1f}|{r.Close:.2f}|{r.RefTP:.2f}|{r.RefSL:.2f}|{why}|')
        lines.append('')
    if len(existing_today): lines += [f'🔒 PRIMARY zaten kilitli: **{existing_today.iloc[0]["Hisse"]}**','']
    elif mode=='primary' and not stale and len(qualified): lines += [f'🔥 PRIMARY: **{qualified.iloc[0]["Hisse"]}**','']
    if summary is not None and len(summary):
        s=summary.iloc[0]; lines += ['## Paper','', f'- Equity: **{s.FinalEquity:,.2f} TL**', f'- Getiri: **%{s.TotalReturn*100:.2f}**', f'- MaxDD: **%{s.MaxDD*100:.2f}**',
            f'- Kapalı/Açık/Bekleyen: **{int(s.ClosedTrades)} / {int(s.OpenTrades)} / {int(s.PendingTrades)}**']
        if pd.notna(s.WR): lines += [f'- WR: **%{s.WR*100:.2f}**', f'- PF: **{s.PF:.3f}**']
    lines += ['', '> PREVIEW kalıcı state yazmaz. PRIMARY yalnız paper test içindir; gerçek emir göndermez.']
    atomic_write_text(REPORT_FILE,'\n'.join(lines)+'\n')


def print_top3(q):
    print('\n🏆 ADAPTİF ML TOP-3')
    print('-'*90)
    for _,r in q.iterrows():
        print(f'✅ #{int(r.Rank)} {r.Hisse} | ML %{r.AdaptiveScore*100:.1f} | fiyat {r.Close:.2f} TL')
        print(f'   📍 Referans TP {r.RefTP:.2f} | SL {r.RefSL:.2f}')
        pos,neg=_xgb_local_contributions(r)
        print('   🧠 AI:', ' | '.join(f'{z["label"]}={_fmt_feature_value(z["feature"],z["value"])}' for z in pos[:3]))
        print('-'*90)


def run(mode: str):
    global ALL, live_model
    persist=(mode=='primary')
    bist_list,_=select_universe()
    ALL,raw_history_cache=build_all_data(bist_list)
    latest_date=pd.Timestamp(ALL['Tarih'].max()).normalize()
    market_days=pd.DatetimeIndex(sorted(pd.to_datetime(ALL['Tarih']).dt.normalize().unique()))
    local_today=datetime.now(TZ).date()
    stale=(latest_date.date()!=local_today)

    live_model,meta,retrained=get_live_model(ALL,latest_date,market_days,persist=persist and not stale)
    today=ALL[ALL['Tarih'].dt.normalize()==latest_date].copy()
    today['AdaptiveScore']=live_model.predict_proba(today[FEATURES].to_numpy(np.float32))[:,1]
    today=today.sort_values('AdaptiveScore',ascending=False).reset_index(drop=True)

    ph=load_primary_signals(); q=qualified_top3(today,ph,latest_date,market_days); print_top3(q)
    existing=ph[ph['SignalDate']==latest_date].copy() if len(ph) else pd.DataFrame()

    trades=summary=None
    if mode=='preview':
        print('\n👀 PREVIEW — hiçbir model/paper/log state dosyası değiştirilmeyecek.')
    else:
        if stale:
            print(f'\n⏸️ PRIMARY YAZILMADI: yerel tarih {local_today}, Yahoo son günlük bar {latest_date.date()}.')
            # Eski state'i yalnız görüntüle; stale günde yeni signal/model state yok.
        else:
            update_watchlist(q,latest_date)
            if len(existing):
                print(f'\n🔒 Bugünün PRIMARY sinyali zaten kilitli: {existing.iloc[0]["Hisse"]}')
            elif len(q):
                ph=add_primary_signal(ph,q.iloc[0],latest_date); existing=ph[ph['SignalDate']==latest_date]
                print(f'\n🔥 PRIMARY KİLİTLENDİ: {q.iloc[0]["Hisse"]}')
            trades,_,summary=persist_paper(ph,raw_history_cache,latest_date)

    make_report(mode,latest_date,retrained,meta,q,existing,trades=trades,summary=summary,stale=stale)
    print(f'\n📄 Rapor: {REPORT_FILE}')
    return 0


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['primary','preview'],default='preview',help='primary=kalıcı paper/state; preview=yalnız görüntüleme')
    return p.parse_args()

if __name__=='__main__':
    try: sys.exit(run(parse_args().mode))
    except Exception as e:
        print(f'\n❌ ÇALIŞMA BAŞARISIZ: {type(e).__name__}: {e}',file=sys.stderr)
        raise
