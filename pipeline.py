# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import importlib.util
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- EDA / ML ekleri ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from pandas.api.types import CategoricalDtype

# ---------------------------------------------------------------------
# Pandas görünüm ayarları
# ---------------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 180)
pd.set_option('display.max_colwidth', 120)

# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------
DATA_PATH = os.path.join("data", "Talent_Academy_Case_DT_2025.xlsx")
RULES_DIR = "rules"  # dış modül kuralları burada

# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------
def UC(txt) -> str:
    """Tüm label/title/legend metinlerini TÜMÜ BÜYÜK olacak şekilde normalize eder (Türkçe 'i/ı' desteği)."""
    if txt is None:
        return ""
    s = str(txt)
    tr = str.maketrans({'i': 'İ', 'ı': 'I'})
    return s.translate(tr).upper()

def _load_py_module(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Modül yüklenemedi: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _compile_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[re.Pattern, str]]:
    compiled: List[Tuple[re.Pattern, str]] = []
    for i, (pat, repl) in enumerate(pairs):
        try:
            compiled.append((re.compile(pat, flags=re.UNICODE), repl))
        except re.error as e:
            raise ValueError(f"Regex derleme hatası (index={i}, pattern={pat}): {e}")
    return compiled

def load_rules_from_dir_py(rules_dir: str) -> Dict:
    common_mod = _load_py_module(os.path.join(rules_dir, 'common.py'))
    tanilar_mod = _load_py_module(os.path.join(rules_dir, 'tanilar.py'))
    alerji_mod  = _load_py_module(os.path.join(rules_dir, 'alerji.py'))
    kronik_mod  = _load_py_module(os.path.join(rules_dir, 'kronik.py'))
    uyg_mod     = _load_py_module(os.path.join(rules_dir, 'uyg_yer.py'))
    noise_mod   = _load_py_module(os.path.join(rules_dir, 'noise.py'))
    site_mod    = _load_py_module(os.path.join(rules_dir, 'site_groups.py'))
    tedavi_mod  = _load_py_module(os.path.join(rules_dir, 'tedavi.py'))

    common_rules = _compile_pairs(getattr(common_mod, 'patterns', []))
    tanilar_rules = _compile_pairs(getattr(tanilar_mod, 'patterns', []))
    alerji_rules  = _compile_pairs(getattr(alerji_mod,  'patterns', []))
    kronik_rules  = _compile_pairs(getattr(kronik_mod,  'patterns', []))
    uyg_rules     = _compile_pairs(getattr(uyg_mod,     'patterns', []))
    tedavi_rules  = _compile_pairs(getattr(tedavi_mod,  'patterns', []))

    tedavi_drop   = set(getattr(noise_mod, 'tedavi_drop_set', []))
    noise_tanilar = set(getattr(noise_mod, 'tanilar_drop_set', []))
    site_groups   = getattr(site_mod, 'groups', [])

    per_column = {
        'KronikHastalik':   {'rules': common_rules + kronik_rules, 'drop': None},
        'Alerji':           {'rules': common_rules + alerji_rules,  'drop': None},
        'Tanilar':          {'rules': common_rules + tanilar_rules, 'drop': noise_tanilar},
        'UygulamaYerleri':  {'rules': common_rules + uyg_rules,     'drop': None},
        'TedaviAdi':        {'rules': common_rules + tedavi_rules,   'drop': tedavi_drop},
    }

    return {'per_column': per_column, 'site_groups': site_groups}

# ---------------------------------------------------------------------
# Keşif/temizlik yardımcıları
# ---------------------------------------------------------------------
EXCLUDE_ID_COLS_FOR_VIZ = {"HastaNo", "HastaTedaviID", "HastaTedaviSeansID"}

def read_and_explore(file_path: str) -> pd.DataFrame:
    print("Veri okunuyor...")
    data = pd.read_excel(file_path)

    print("\n>>> İlk 5 satır:")
    print(data.head())

    print("\n>>> Boyut:", data.shape)

    print("\n>>> Bilgi:")
    data.info()

    # Describe özetleri
    num_desc = data.describe(include=[np.number]).T
    obj_desc = data.describe(include=['object']).T
    print("\n>>> Tanımlayıcı istatistikler (sayısal):")
    print(num_desc)
    print("\n>>> Tanımlayıcı istatistikler (kategorik):")
    print(obj_desc)

    # Boş değer özeti (yalnızca NA içeren kolonlar)
    na = data.isna().sum()
    na = na[na > 0].sort_values(ascending=False)
    print("\n>>> Boş değer sayıları:")
    if na.empty:
        print(" - (Hiç boş değer yok)")
    else:
        for k, v in na.items():
            print(f" - {k}: {v}")

    # Boş değer özetinden SONRA EDA görselleri
    try:
        run_eda_visuals(data)
        print(">>> EDA görselleri kaydedildi (reports/figures).")
    except Exception as e:
        print(f"[WARN] EDA görselleştirme sırasında hata: {e}")

    return data


def clean_and_int(series: pd.Series) -> pd.Series:
    """'15 Seans' → 15, '20 Dakika' → 20; NA destekli tam sayı döner (Int64)."""
    return (series.astype(str)
                 .str.replace(" Seans", "", regex=False)
                 .str.replace(" Dakika", "", regex=False)
                 .str.strip()
                 .replace({"": np.nan})
                 .astype(float)
                 .astype('Int64'))

def convert_to_categorical(data: pd.DataFrame, threshold: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """Düşük kardinaliteli object/string sütunları kategorik yap (NaN'ı metne çevirmeden)."""
    cat_cols: List[str] = []
    for col in data.columns:
        dtype = data[col].dtype
        if isinstance(dtype, CategoricalDtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            nunq = data[col].nunique(dropna=True)
            if nunq <= threshold:
                s = data[col].astype("string")  # StringDtype -> NaN korunur
                s = s.str.strip()
                s = s.mask(s.str.len() == 0)
                s = s.mask(s.str.lower().isin(["nan", "none", "null", "na"]))
                data[col] = s.astype("category")
                cat_cols.append(col)
    # Burada print yok; raporda yazdıracağız
    return data, cat_cols

# ---------------------------------------------------------------------
# Normalize edici çekirdek fonksiyon
# ---------------------------------------------------------------------
def normalize_multi_value(series: pd.Series,
                          sep: str = ',',
                          canon_rules: Optional[List[Tuple[re.Pattern, str]]] = None,
                          drop_set: Optional[set] = None) -> pd.Series:
    TR_UP = str.maketrans({'i': 'İ', 'ı': 'I'})

    def _norm_token(token: str) -> str:
        s = str(token).translate(TR_UP).upper()
        s = re.sub(r"[^A-Z0-9ÇĞİÖŞÜÂÊÎÔÛ\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if canon_rules:
            for pat, repl in canon_rules:
                s = pat.sub(repl, s)
                s = re.sub(r"\s+", " ", s).strip()
        return s

    def _clean_cell(cell):
        if pd.isna(cell):
            return np.nan
        txt = str(cell).strip()
        if txt.lower() in {"nan", "none", ""}:
            return np.nan
        parts = txt.split(sep)
        seen, out = set(), []
        for p in parts:
            n = _norm_token(p)
            if not n:
                continue
            if drop_set and n in drop_set:
                continue
            if n not in seen:
                seen.add(n)
                out.append(n)
        return ", ".join(out) if out else np.nan

    return series.apply(_clean_cell)

# ---------------------------------------------------------------------
# Uygulama yerlerini üst gruplara map etme
# ---------------------------------------------------------------------
def map_sites_multi(series: pd.Series, groups: List[Dict], sep: str = ',') -> pd.Series:
    compiled = [(re.compile(g['pattern'], flags=re.UNICODE), g['label']) for g in groups]

    def _map_cell(cell):
        if pd.isna(cell):
            return np.nan
        toks = [t.strip() for t in str(cell).split(sep)]
        out, seen = [], set()
        for t in toks:
            label = None
            for pat, lab in compiled:
                if pat.search(t):
                    label = lab
                    break
            label = label or t
            if label not in seen:
                seen.add(label)
                out.append(label)
        return ", ".join(out) if out else np.nan

    return series.apply(_map_cell)

# ---------------------------------------------------------------------
# Token bazlı value_counts (çok-değerli hücreler için)
# ---------------------------------------------------------------------
def token_value_counts(series: pd.Series, sep: str = ',') -> pd.Series:
    return (
        series.dropna()
              .str.split(rf'\s*{re.escape(sep)}\s*')
              .explode()
              .value_counts()
    )

# ---------------------------------------------------------------------
# TedaviAdi ön-temizlik + yinelenen kelime sadeleştirme
# ---------------------------------------------------------------------
def _pre_clean_tedavi(s: pd.Series) -> pd.Series:
    s2 = (s.astype(str)
            .str.replace(r'[\+\-/]+', ',', regex=True)
            .str.replace(r'\s*,\s*', ',', regex=True)
            .str.strip(', '))
    return s2

# ---------------------------------------------------------------------
# ID üretimi
# ---------------------------------------------------------------------
def _slugify_tedavi(x) -> str:
    import re as _re
    import pandas as _pd
    if _pd.isna(x): return "na"
    tr_map = str.maketrans({'Ç':'C','Ğ':'G','İ':'I','I':'I','Ö':'O','Ş':'S','Ü':'U',
                            'ç':'c','ğ':'g','ı':'i','i':'i','ö':'o','ş':'s','ü':'u'})
    s = str(x).translate(tr_map).lower()
    s = _re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    return s or "na"

def add_ids(df: pd.DataFrame) -> pd.DataFrame:
    hasta_str = df['HastaNo'].astype('Int64').astype('string').fillna('NA')
    tedavi_src = df['TedaviAdi_cleaned'] if 'TedaviAdi_cleaned' in df.columns else df['TedaviAdi']
    tedavi_slug = tedavi_src.apply(_slugify_tedavi)

    df['HastaTedaviID'] = hasta_str + '::' + tedavi_slug

    if 'SeansIndex' not in df.columns:
        df = df.sort_index()
        df['SeansIndex'] = df.groupby(['HastaNo', tedavi_src.name], dropna=False).cumcount() + 1

    df['HastaTedaviSeansID'] = df['HastaTedaviID'] + '#' + df['SeansIndex'].astype(int).astype(str).str.zfill(3)
    return df

# ---------------------------------------------------------------------
# Feature mühendisliği yardımcıları
# ---------------------------------------------------------------------
def _tokenize(cell: str, sep: str = ',') -> list[str]:
    if pd.isna(cell):
        return []
    return [t.strip() for t in str(cell).split(sep) if str(t).strip()]

def _slug_token(t: str) -> str:
    tr = str.maketrans({'Ç':'C','Ğ':'G','İ':'I','I':'I','Ö':'O','Ş':'S','Ü':'U',
                        'ç':'c','ğ':'g','ı':'i','i':'i','ö':'o','ş':'s','ü':'u'})
    s = t.translate(tr).lower()
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s or 'na'

def add_primary_and_counts(df: pd.DataFrame, cols: list[str], sep: str = ',') -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        toks = df[c].map(lambda s: _tokenize(s, sep))
        df[f"{c}__primary"] = toks.map(lambda lst: lst[0] if lst else np.nan)
        df[f"{c}__count"]   = toks.map(lambda lst: len(set(lst)))
    return df

def add_multi_hot(df: pd.DataFrame,
                  col: str,
                  prefix: str,
                  sep: str = ',',
                  top_n: int | None = 20,
                  min_freq: int | None = None,
                  make_other: bool = True,
                  dtype='Int8') -> tuple[pd.DataFrame, pd.Series]:

    if col not in df.columns:
        return df, pd.Series(dtype='int64')

    counts = token_value_counts(df[col], sep=sep)

    if min_freq is not None:
        keep = list(counts[counts >= min_freq].index)
    elif top_n is not None:
        keep = list(counts.head(top_n).index)
    else:
        keep = list(counts.index)

    keep_set = set(keep)
    patterns = {
        t: re.compile(
            rf'(?:^|\s*{re.escape(sep)}\s*){re.escape(t)}(?:\s*{re.escape(sep)}\s*|$)',
            flags=re.UNICODE
        )
        for t in keep
    }

    base = df[col].fillna('')
    for t in keep:
        slug = _slug_token(t)
        df[f"{prefix}__{slug}"] = base.str.contains(patterns[t], na=False).astype(dtype)

    if make_other:
        def _has_other(s):
            toks = set(_tokenize(s, sep))
            return int(bool(toks - keep_set))
        df[f"{prefix}__other"] = df[col].map(_has_other).astype(dtype)

    df[f"{prefix}__n"] = df[col].map(lambda s: len(set(_tokenize(s, sep)))).astype('Int16')
    return df, counts

# ---------------------------------------------------------------------
# Tedavi düzeyi özet/metric'ler
# ---------------------------------------------------------------------
def add_session_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_tx = (df.groupby("HastaTedaviID", as_index=False)
               .agg(**{
                   "SeansObserved": ("SeansIndex", "max"),
                   **({"SeansPlanned": ("TedaviSuresi(Seans)", "max")} if "TedaviSuresi(Seans)" in df.columns else {}),
                   **({"MeanUygSuresi": ("UygulamaSuresi(Dakika)", "mean"),
                      "SumUygSuresi":  ("UygulamaSuresi(Dakika)", "sum")} if "UygulamaSuresi(Dakika)" in df.columns else {}),
                   **({"PrimarySite": ("UygulamaYerleri_grouped__primary", "first")} if "UygulamaYerleri_grouped__primary" in df.columns else {}),
                   **({"SiteCount": ("UygulamaYerleri_grouped__count", "max")} if "UygulamaYerleri_grouped__count" in df.columns else {}),
                   **({"PrimaryDx": ("Tanilar_cleaned__primary", "first")} if "Tanilar_cleaned__primary" in df.columns else {}),
                   **({"DxCount": ("Tanilar_cleaned__count", "max")} if "Tanilar_cleaned__count" in df.columns else {}),
                   **({"PrimaryTx": ("TedaviAdi_cleaned__primary", "first")} if "TedaviAdi_cleaned__primary" in df.columns else {}),
                   **({"TxCount": ("TedaviAdi_cleaned__count", "max")} if "TedaviAdi_cleaned__count" in df.columns else {}),
               }))

    if "SeansPlanned" in by_tx.columns:
        by_tx["SeansKalan"] = (by_tx["SeansPlanned"] - by_tx["SeansObserved"]).clip(lower=0)
        by_tx["Coverage"]   = (by_tx["SeansObserved"] / by_tx["SeansPlanned"]).round(2)
    else:
        by_tx["SeansKalan"] = pd.NA
        by_tx["Coverage"]   = pd.NA

    df = df.merge(by_tx, on="HastaTedaviID", how="left", suffixes=("", "_byTX"))

    df["IsFirstSeans"]   = (df["SeansIndex"] == 1).astype("Int8")
    df["IsLastSeansObs"] = (df["SeansIndex"] == df["SeansObserved"]).astype("Int8")

    if "SeansPlanned" in df.columns:
        cond = df["SeansPlanned"].notna()
        df["IsPlannedLast"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        df.loc[cond, "IsPlannedLast"] = (df.loc[cond, "SeansIndex"] == df.loc[cond, "SeansPlanned"]).astype("Int8")
    else:
        df["IsPlannedLast"] = pd.Series(pd.NA, index=df.index, dtype="Int8")

    return df, by_tx

# ---------------------------------------------------------------------
# EDA görselleri (TÜM LABEL/TITLE/LEGEND UPPERCASE)
# ---------------------------------------------------------------------
def run_eda_visuals(data: pd.DataFrame, outdir: str = os.path.join("reports", "figures")) -> None:
    """
    - UYRUK: YATAY BAR (TOP10 + 'DİĞER')
    - TEDAVİSURESİ: SAYISAL HİSTOGRAM (TAM SAYI BİN), ORT/MEDYAN ÇİZGİLERİ
    - TEDAVİADİ: YATAY BAR (TOP30)
    - CİNSİYET: NAN DAHİL, HER BAR FARKLI RENK, YÜZDE ETİKETLERİ
    - UYGULAMAYERLERİ: TOKEN'LANMIŞ TOP30 YATAY BAR
    - SEABORN: SAYISAL KORELASYON ISI HARİTASI + YAŞ↔SEANS SCATTER
    """
    os.makedirs(outdir, exist_ok=True)

    plt.rcParams.update({
        "axes.titlesize": 16, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12
    })
    sns.set_theme(style="whitegrid", context="talk")

    # -------- yardımcılar --------
    def _clean_series(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.strip()
                .replace({"": np.nan, "nan": np.nan}))

    def _vc_dropna(s: pd.Series) -> pd.Series:
        return _clean_series(s).dropna().value_counts()

    def _vc_with_other(vc: pd.Series, top_n: int) -> pd.Series:
        if len(vc) <= top_n:
            return vc
        head = vc.head(top_n).copy()
        head.loc[UC("Diğer")] = vc.iloc[top_n:].sum()
        return head

    def _save(fig, path):
        try:
            fig.tight_layout()
        except Exception:
            fig.subplots_adjust(bottom=0.22)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def _annotate_hbars(ax, fmt="{:d}"):
        for p in ax.patches:
            w = p.get_width()
            if w > 0:
                ax.text(w + max(1, 0.01*w), p.get_y() + p.get_height()/2,
                        fmt.format(int(round(w))), va="center", fontsize=11)

    # -------- YAŞ (HİSTOGRAM) --------
    if "Yas" in data.columns:
        s = pd.to_numeric(data["Yas"], errors="coerce").dropna()
        if len(s) > 0:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.hist(s, bins=30, edgecolor="black", alpha=0.75)
            ax.set_title(UC("Histogram - Yaş"))
            ax.set_xlabel(UC("Yaş"))
            ax.set_ylabel(UC("Frekans"))
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            _save(fig, os.path.join(outdir, "hist_Yas.png"))

    # -------- TEDAVİ SÜRESİ (SEANS) HİSTOGRAM --------
    seans_num = None
    if "TedaviSuresi(Seans)" in data.columns:
        seans_num = pd.to_numeric(data["TedaviSuresi(Seans)"], errors="coerce")
    elif "TedaviSuresi" in data.columns:
        seans_num = pd.to_numeric(clean_and_int(data["TedaviSuresi"]).astype("Float64"), errors="coerce")
    if seans_num is not None:
        s = seans_num.dropna().astype(int)
        if len(s) > 0:
            bins = np.arange(s.min()-0.5, s.max()+1.5, 1)
            fig, ax = plt.subplots(figsize=(12, 6.5))
            ax.hist(s, bins=bins, edgecolor="black", alpha=0.8)
            ax.set_xticks(np.arange(s.min(), s.max()+1, max(1, (s.max()-s.min())//15 or 1)))
            ax.set_title(UC("Seans Sayısı Dağılımı"))
            ax.set_xlabel(UC("Seans"))
            ax.set_ylabel(UC("Adet"))
            mean_v = s.mean(); med_v = s.median()
            ax.axvline(mean_v, color="#2e86c1", linestyle="--", linewidth=2, label=UC(f"Ortalama: {mean_v:.1f}"))
            ax.axvline(med_v, color="#c0392b", linestyle="-.", linewidth=2, label=UC(f"Medyan: {med_v:.0f}"))
            ax.legend(loc="upper right", title=UC("Özet"))
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            _save(fig, os.path.join(outdir, "hist_TedaviSuresiSeans.png"))

    # -------- UYGULAMA SÜRESİ DONUT --------
    if "UygulamaSuresi" in data.columns:
        vc_all = _vc_dropna(data["UygulamaSuresi"])
        vc = _vc_with_other(vc_all, top_n=6)
        if len(vc) > 0:
            fig, ax = plt.subplots(figsize=(14, 5))
            wedges, *_ = ax.pie(vc.values, labels=None, startangle=90,
                                autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
                                pctdistance=0.78, textprops={"fontsize": 12})
            centre_circle = plt.Circle((0, 0), 0.55, fc="white")
            fig.gca().add_artist(centre_circle)
            ax.set_title(UC("Donut - Uygulama Süresi (Top6 + Diğer)"))
            legend_labels = [UC(x) for x in vc.index.tolist()]
            lg = ax.legend(wedges, legend_labels, title=UC("Uygulama Süresi"),
                           loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)
            lg.set_title(UC("Uygulama Süresi"))
            _save(fig, os.path.join(outdir, "donut_UygulamaSuresi.png"))

    # -------- UYGULAMA YERLERİ (TOP30 YATAY BAR) --------
    if "UygulamaYerleri" in data.columns:
        tok_counts = token_value_counts(_clean_series(data["UygulamaYerleri"]))
        vc = tok_counts.head(30).sort_values()
        if len(vc) > 0:
            fig, ax = plt.subplots(figsize=(10, 12))
            y = np.arange(len(vc))
            ax.barh(y, vc.values)
            ax.set_yticks(y)
            ax.set_yticklabels([UC(x) for x in vc.index], fontsize=11)
            ax.set_xlabel(UC("Adet"))
            ax.set_title(UC("Kategori Dağılımı - Uygulama Yerleri (Top30)"))
            _annotate_hbars(ax)
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            _save(fig, os.path.join(outdir, "hbar_UygulamaYerleri.png"))

    # -------- UYRUK (TOP10 + DİĞER) --------
    if "Uyruk" in data.columns:
        vc_all = _vc_dropna(data["Uyruk"])
        vc = _vc_with_other(vc_all, top_n=10).sort_values()
        if len(vc) > 0:
            total = vc.sum()
            fig, ax = plt.subplots(figsize=(12, 8))
            y = np.arange(len(vc))
            ax.barh(y, vc.values)
            ax.set_yticks(y)
            ax.set_yticklabels([UC(x) for x in vc.index])
            ax.set_xlabel(UC("Adet"))
            ax.set_title(UC("Uyruk - Top10 + Diğer (Yatay Bar)"))
            for i, v in enumerate(vc.values):
                ax.text(v + max(1, 0.01*v), i, UC(f"{int(v)}  ({v/total*100:.1f}%)"), va="center", fontsize=11)
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            _save(fig, os.path.join(outdir, "barh_Uyruk.png"))

    # -------- CİNSİYET (YÜZDE BAR, NAN DAHİL) --------
    if "Cinsiyet" in data.columns:
        s = _clean_series(data["Cinsiyet"])
        vc = s.fillna("NaN").value_counts()
        if len(vc) > 0:
            total = vc.sum()
            fig, ax = plt.subplots(figsize=(14, 4.8))
            colors = plt.cm.tab10(np.linspace(0, 1, len(vc)))
            cats = [UC(x) for x in vc.index.tolist()]
            bars = ax.bar(cats, (vc/total*100).values, color=colors)
            ax.set_ylim(0, 100)
            ax.set_ylabel(UC("%"))
            ax.set_title(UC("Cinsiyet - Yüzde Dağılım (NaN Dahil)"))
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            for b, p in zip(bars, (vc/total*100).values):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+2, UC(f"{p:.1f}%"), ha="center", fontsize=12)
            _save(fig, os.path.join(outdir, "barpct_Cinsiyet_withNaN.png"))

    # -------- KAN GRUBU (ABO × RH) --------
    if "KanGrubu" in data.columns:
        raw = (data["KanGrubu"].astype(str).str.replace(r"\s+", "", regex=True).str.strip())
        def _parse_abo_rh(x: str):
            m = re.match(r"^(0|O|A|B|AB)Rh([+-])$", x)
            if not m: return None
            return m.group(1).replace("0", "O"), m.group(2)
        parsed = raw.map(_parse_abo_rh).dropna()
        if len(parsed) > 0:
            abo_rh = pd.DataFrame(parsed.tolist(), columns=["ABO", "Rh"])
            tbl = abo_rh.value_counts().unstack(fill_value=0).reindex(index=["O","A","B","AB"], fill_value=0)
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(tbl.index)); w = 0.35
            ax.bar(x-w/2, tbl.get("+", pd.Series(0, index=tbl.index)), width=w, label=UC("Rh+"))
            ax.bar(x+w/2, tbl.get("-", pd.Series(0, index=tbl.index)), width=w, label=UC("Rh-"))
            ax.set_xticks(x); ax.set_xticklabels([UC(ix) for ix in tbl.index])
            ax.set_ylabel(UC("Adet"))
            ax.set_title(UC("Kan Grubu - ABO × Rh"))
            leg = ax.legend(title=UC("Rh"))
            leg.set_title(UC("Rh"))
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            _save(fig, os.path.join(outdir, "grouped_KanGrubu_ABO_Rh.png"))

    # ======================
    # SEABORN EKLER
    # ======================

    # --- KORELASYON ISI HARİTASI ---
    try:
        # 1) Sayısala çevir
        if "TedaviSuresi(Seans)" in data.columns:
            seans = pd.to_numeric(data["TedaviSuresi(Seans)"], errors="coerce")
        elif "TedaviSuresi" in data.columns:
            seans = pd.to_numeric(clean_and_int(data["TedaviSuresi"]).astype("Float64"), errors="coerce")
        else:
            seans = pd.Series(np.nan, index=data.index)

        if "UygulamaSuresi(Dakika)" in data.columns:
            dakika = pd.to_numeric(data["UygulamaSuresi(Dakika)"], errors="coerce")
        elif "UygulamaSuresi" in data.columns:
            dakika = pd.to_numeric(clean_and_int(data["UygulamaSuresi"]).astype("Float64"), errors="coerce")
        else:
            dakika = pd.Series(np.nan, index=data.index)

        yas = pd.to_numeric(data["Yas"], errors="coerce") if "Yas" in data.columns else pd.Series(np.nan,
                                                                                                  index=data.index)

        num_df = pd.DataFrame({
            "TedaviSuresi(Seans)": seans,
            "UygulamaSuresi(Dakika)": dakika,
            "Yas": yas
        })

        corr = num_df.corr(method="pearson")

        # 2) Hedefi ilk sıraya koy ve kısa eksen etiketleri hazırla
        order = [c for c in ["TedaviSuresi(Seans)", "UygulamaSuresi(Dakika)", "Yas"] if c in corr.index]
        corr = corr.loc[order, order]
        short_labels_map = {
            "TedaviSuresi(Seans)": UC("Seans"),
            "UygulamaSuresi(Dakika)": UC("Dakika"),
            "Yas": UC("Yaş")
        }
        disp_labels = [short_labels_map.get(c, UC(c)) for c in order]

        # 3) Çizim (geniş figür + marjin + döndürülmüş x etiketleri)
        if corr.shape[0] >= 2:
            fig, ax = plt.subplots(figsize=(9.6, 7.0))
            hm = sns.heatmap(
                corr,
                ax=ax,
                cmap="coolwarm",
                center=0,
                vmin=-1, vmax=1,
                annot=True, fmt=".2f",
                linewidths=0.6, linecolor="white",
                square=False,  # kare zorlamayı kaldır
                cbar_kws={"shrink": .85, "label": UC("Pearson r")},
                annot_kws={"size": 11}
            )

            # Eksen etiketleri
            ax.set_xticklabels(disp_labels, rotation=35, ha="right")
            ax.set_yticklabels(disp_labels, rotation=0)
            ax.tick_params(axis='x', pad=6, labelsize=11)
            ax.tick_params(axis='y', labelsize=11)

            # Başlığı iki satıra böl + boşluk
            ax.set_title(UC("Korelasyon Isı Haritası\n(Hedef: Seans)"), pad=12)

            # Marjinleri genişlet (etiketler kesilmesin)
            fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.28)

            _save(fig, os.path.join(outdir, "heatmap_corr_numeric.png"))
        else:
            print("[WARN] Isı haritası için yeterli sayısal kolon bulunamadı.")
    except Exception as e:
        print(f"[WARN] Korelasyon ısı haritası üretilemedi: {e}")

    # --- SCATTER: YAŞ vs TEDAVİSÜRESİ(SEANS) ---
    try:
        if "Yas" in data.columns:
            y = pd.to_numeric(data["Yas"], errors="coerce")
            if "TedaviSuresi(Seans)" in data.columns:
                s = pd.to_numeric(data["TedaviSuresi(Seans)"], errors="coerce")
            elif "TedaviSuresi" in data.columns:
                s = pd.to_numeric(clean_and_int(data["TedaviSuresi"]).astype("Float64"), errors="coerce")
            else:
                s = None

            if s is not None:
                m = y.notna() & s.notna()
                if m.any():
                    fig, ax = plt.subplots(figsize=(7.8, 5.8))
                    sns.scatterplot(x=y[m], y=s[m], ax=ax, alpha=0.35)
                    sns.regplot(x=y[m], y=s[m], ax=ax, scatter=False, ci=None, line_kws={"linewidth": 2})
                    ax.set_xlabel(UC("Yaş"))
                    ax.set_ylabel(UC("Seans"))
                    ax.set_title(UC("Yaş ↔ TedaviSuresi(Seans) (Seaborn Scatter + Trend)"))
                    ax.grid(True, linestyle=":", alpha=0.5)
                    _save(fig, os.path.join(outdir, "scatter_Yas_vs_Seans.png"))
    except Exception as e:
        print(f"[WARN] Scatter üretilemedi: {e}")

    # -------- TEDAVİADİ (TOP30 YATAY BAR) --------
    if "TedaviAdi" in data.columns:
        vc = _vc_dropna(data["TedaviAdi"]).head(30).sort_values()
        if len(vc) > 0:
            fig, ax = plt.subplots(figsize=(10, 12))
            y = np.arange(len(vc))
            ax.barh(y, vc.values)
            ax.set_yticks(y); ax.set_yticklabels([UC(x) for x in vc.index], fontsize=11)
            ax.set_xlabel(UC("Adet"))
            ax.set_title(UC("Tedavi Adı - Top30 (Yatay Bar)"))
            _annotate_hbars(ax)
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            _save(fig, os.path.join(outdir, "barh_TedaviAdi.png"))

# ---------------------------------------------------------------------
# MODELLEMEYE HAZIR TABLO: KNN ile eksik doldurma (sayısal + kategorik)
# ---------------------------------------------------------------------
def _clean_kan_grubu(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()

def _strip_and_nanify(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.mask(s.str.len() == 0)
    s = s.mask(s.str.lower().isin(["nan", "none", "null", "na"]))
    return s

def _encode_cats_to_codes_for_knn(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[int, str]], Dict[str, Dict[str, int]]]:
    code_to_label: Dict[str, Dict[int, str]] = {}
    label_to_code: Dict[str, Dict[str, int]] = {}
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype("string")
        cats = pd.Series(s.dropna().unique()).sort_values(kind="mergesort")
        l2c = {lab: i for i, lab in enumerate(cats.tolist())}
        c2l = {i: lab for lab, i in l2c.items()}
        label_to_code[c] = l2c
        code_to_label[c] = c2l
        out[c] = s.map(l2c).astype("Float64")
    return out, code_to_label, label_to_code

def _decode_codes_to_cats(df_codes: pd.DataFrame, code_to_label: Dict[str, Dict[int, str]], cols: List[str]) -> pd.DataFrame:
    out = df_codes.copy()
    for c in cols:
        if c not in out.columns or c not in code_to_label:
            continue
        if out[c].isna().all():
            continue
        arr = out[c].astype(float).round().astype("Int64")
        valid_max = max(code_to_label[c].keys()) if len(code_to_label[c]) else -1
        if valid_max >= 0:
            arr = arr.clip(lower=0, upper=valid_max)
        out[c] = arr.map(code_to_label[c]).astype("string")
    return out

def build_model_ready(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Nihai tablo:
      - Eksikler: KNNImputer. Kategorikler geçici kodlanır, sonrası etikete döner.
      - 'Yas' önce KNN ile doldurulur, SONRA StandardScaler ile ölçeklenir.
      - OHE: Cinsiyet/KanGrubu/Uyruk/Bolum (drop_first=True).
      - Multi-hot: DX__/CHR__/ALG__/SITE__/TX__ 1/0 sütunları (__n hariç).
      - Rapor: yalnızca gerçekten doldurulan kolonlar ve sayıları.
    """
    data = df.copy()

    # Dakika kolonu standardı
    if "UygulamaSuresi(Dakika)" not in data.columns:
        if "Tedavi Süresi(Dakika)" in data.columns:
            data.rename(columns={"Tedavi Süresi(Dakika)": "UygulamaSuresi(Dakika)"}, inplace=True)
        else:
            data["UygulamaSuresi(Dakika)"] = pd.NA

    # Kategorik baz kolonlar
    base_cat_cols = [nm for nm in ["Cinsiyet", "KanGrubu", "Uyruk", "Bolum"] if nm in data.columns]
    if "KanGrubu" in base_cat_cols:
        data["KanGrubu"] = _clean_kan_grubu(data["KanGrubu"])

    # KNN öncesi kategorikleri normalize et
    if base_cat_cols:
        for c in base_cat_cols:
            data[c] = _strip_and_nanify(data[c])

    # KNN sayısallar
    num_cols_for_knn = [c for c in ["TedaviSuresi(Seans)", "UygulamaSuresi(Dakika)", "Yas"] if c in data.columns]
    for c in num_cols_for_knn:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # İmputasyon metrikleri
    knn_cols = num_cols_for_knn + base_cat_cols
    before_na = {c: int(data[c].isna().sum()) for c in knn_cols}

    # Encode → KNN
    data_enc, code_to_label, label_to_code = _encode_cats_to_codes_for_knn(data, base_cat_cols)
    knn_df = data_enc[knn_cols].copy()

    if knn_df.shape[1] > 0 and (~knn_df.isna()).any(axis=None):
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        imputed = imputer.fit_transform(knn_df.astype(float))
        knn_imputed = pd.DataFrame(imputed, columns=knn_cols, index=knn_df.index)
    else:
        knn_imputed = knn_df.copy()

    after_knn_na = {c: int(knn_imputed[c].isna().sum()) for c in knn_cols}
    residual = {c: n for c, n in after_knn_na.items() if n > 0}
    if residual:
        raise ValueError(f"KNN sonrası bazı kolonlarda NA kaldı: {residual}")

    # Decode geri yaz
    back = _decode_codes_to_cats(knn_imputed, code_to_label, base_cat_cols)
    for c in num_cols_for_knn:
        data[c] = back[c].astype(float)
    for c in base_cat_cols:
        data[c] = back[c].astype("string")

    # Yaş ölçekle
    yas_scaled = False
    if "Yas" in data.columns:
        scaler = StandardScaler()
        data["Yas"] = scaler.fit_transform(data[["Yas"]])
        yas_scaled = True

    # One-hot
    if base_cat_cols:
        cats_for_ohe = data[base_cat_cols].apply(lambda s: s.astype("string").str.strip())
        ohe_df = pd.get_dummies(
            cats_for_ohe,
            prefix=base_cat_cols,
            prefix_sep="_",
            drop_first=True,
            dtype="Int8",
            dummy_na=False
        )
        ohe_counts = {base: sum(col.startswith(f"{base}_") for col in ohe_df.columns) for base in base_cat_cols}
        order_bases = ["Cinsiyet", "KanGrubu", "Uyruk", "Bolum"]
        ohe_cols_ordered = [col for base in order_bases for col in ohe_df.columns if col.startswith(f"{base}_")]
        ohe_df = ohe_df.reindex(columns=ohe_cols_ordered)
    else:
        ohe_df = pd.DataFrame(index=data.index)
        ohe_counts = {}

    # Multi-hot (1/0, __n hariç)
    multi_hot_cols = [
        c for c in data.columns
        if re.match(r'^(DX|CHR|ALG|SITE|TX)__', c) and not c.endswith("__n")
    ]
    prefixes = ["DX__", "CHR__", "ALG__", "SITE__", "TX__"]
    mh_counts = {p: sum(col.startswith(p) for col in multi_hot_cols) for p in prefixes}

    # Nihai tablo
    base_needed = ["HastaTedaviSeansID", "TedaviSuresi(Seans)", "UygulamaSuresi(Dakika)", "Yas"]
    base_df = data.reindex(columns=[c for c in base_needed if c in data.columns])
    final_df = pd.concat([base_df, ohe_df, data[multi_hot_cols]], axis=1)

    # Rapor
    imputed_numeric_used = [c for c in num_cols_for_knn if before_na.get(c, 0) > 0]
    imputed_categorical_used = [c for c in base_cat_cols if before_na.get(c, 0) > 0]
    filled_by_knn = {c: before_na[c] for c in (imputed_numeric_used + imputed_categorical_used)}
    total_filled = int(sum(filled_by_knn.values()))
    model_ready_total_na = int(final_df.isna().sum().sum())

    report = {
        "categoricals_knn_imputed_used": imputed_categorical_used,
        "numerics_knn_imputed_used": imputed_numeric_used,
        "filled_by_knn": filled_by_knn,
        "total_filled_by_knn": total_filled,
        "yas_scaled": yas_scaled,
        "onehot_bases": base_cat_cols,
        "onehot_counts": ohe_df.shape[1] and {base: sum(col.startswith(f"{base}_") for col in ohe_df.columns) for base in base_cat_cols} or {},
        "onehot_drop_first": True,
        "multi_hot_counts": mh_counts,
        "model_ready_total_na": model_ready_total_na,
    }
    return final_df, report

# ---------------------------------------------------------------------
# Konsol raporu
# ---------------------------------------------------------------------
def _fmt_counts(d: Dict[str, int]) -> str:
    return ", ".join([f"{k}={v}" for k, v in d.items()]) if d else "(yok)"

def print_processing_report(cat_cols: List[str], report: dict, multihot_details: List[dict]) -> None:
    print("\n>>> Kategoriğe çevrilen sütunlar:")
    print(" - " + (", ".join(cat_cols) if cat_cols else "(Yok)"))

    print("\n>>> Eksik değer doldurma (KNNImputer):")
    num_used = report.get("numerics_knn_imputed_used", [])
    cat_used = report.get("categoricals_knn_imputed_used", [])
    print(f" - Doldurulan sayısal kolonlar: {', '.join(num_used) if num_used else '(yok)'}")
    print(f" - Doldurulan kategorik kolonlar: {', '.join(cat_used) if cat_used else '(yok)'}")
    filled = report.get("filled_by_knn", {})
    print(f" - Doldurulan hücre sayıları: {_fmt_counts(filled)}")
    print(f" - Toplam doldurulan hücre: {report.get('total_filled_by_knn', 0)}")

    print("\n>>> Dönüşümler:")
    yas_scaled = report.get("yas_scaled", False)
    print(f" - Yas ölçeklendi: {'Evet' if yas_scaled else 'Hayır'}")
    ohe_bases = report.get("onehot_bases", [])
    ohe_counts = report.get("onehot_counts", {})
    if ohe_bases:
        ohe_line = ", ".join([f"{b}={ohe_counts.get(b, 0)}" for b in ohe_bases])
        print(f" - One-hot (drop_first=True): {ohe_line}")
    else:
        print(" - One-hot: (yok)")
    mh_counts = report.get("multi_hot_counts", {})
    if mh_counts:
        mh_line = ", ".join([f"{k}={v}" for k, v in mh_counts.items()])
        print(f" - Multi-hot: {mh_line}")
    else:
        print(" - Multi-hot: (yok)")

    total_na = report.get("model_ready_total_na", None)
    if total_na is not None:
        print(f"\n>>> Modellemeye hazır tabloda toplam NA: {total_na}")
        if total_na == 0:
            print(">>> Boş değer kontrolü: PAS (hiç NA yok)")
        else:
            print(">>> Boş değer kontrolü: DİKKAT (NA mevcut)")

# ---------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------
def run_pipeline(data_path: str, rules_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    df = read_and_explore(data_path)

    # Süre sütunlarını sayısala çevir
    if "TedaviSuresi" in df.columns:
        df["TedaviSuresi(Seans)"] = clean_and_int(df["TedaviSuresi"])
        df.drop(columns=["TedaviSuresi"], inplace=True)
    if "UygulamaSuresi" in df.columns:
        df["UygulamaSuresi(Dakika)"] = clean_and_int(df["UygulamaSuresi"])
        df.drop(columns=["UygulamaSuresi"], inplace=True)

    # Düşük kardinaliteli object sütunları kategorik yap
    df, cat_cols = convert_to_categorical(df, threshold=10)

    # Kuralları yükle
    rules = load_rules_from_dir_py(rules_dir)
    per_col = rules['per_column']
    site_groups = rules['site_groups']

    # Çok-değerli metin sütunlarını normalize et
    if "TedaviAdi" in df.columns:
        df["TedaviAdi"] = _pre_clean_tedavi(df["TedaviAdi"])

    target_cols = ["KronikHastalik", "Alerji", "Tanilar", "UygulamaYerleri", "TedaviAdi"]
    for c in target_cols:
        if c not in df.columns:
            continue
        cfg = per_col.get(c, {'rules': [], 'drop': None})
        df[f"{c}_cleaned"] = normalize_multi_value(
            df[c],
            canon_rules=cfg['rules'],
            drop_set=cfg['drop']
        )

    # Yinelenen kelimeleri sadeleştir (TedaviAdi_cleaned içinde)
    if "TedaviAdi_cleaned" in df.columns:
        df["TedaviAdi_cleaned"] = df["TedaviAdi_cleaned"].str.replace(
            r'\b([A-ZÇĞİÖŞÜ]+)(?:\s+\1)+\b', r'\1', regex=True
        )

    # Uygulama yerlerini üst-gruplara map et
    if "UygulamaYerleri_cleaned" in df.columns:
        df["UygulamaYerleri_grouped"] = map_sites_multi(df["UygulamaYerleri_cleaned"], site_groups)

    return df, cat_cols

# ---------------------------------------------------------------------
# Çalıştır
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df, cat_cols = run_pipeline(DATA_PATH, RULES_DIR)

    # ID'ler
    df = add_ids(df)

    # Primary & count kolonları
    feature_cols = []
    for c in ["Tanilar_cleaned", "KronikHastalik_cleaned", "Alerji_cleaned", "UygulamaYerleri_grouped", "TedaviAdi_cleaned"]:
        if c in df.columns:
            feature_cols.append(c)
    df = add_primary_and_counts(df, feature_cols, sep=',')

    # Tedavi düzeyi özet metrikler
    df, by_tx = add_session_features(df)

    # Çok-etiketli sütunlar için eksiklik bayrakları
    for src, pfx in [
        ("Tanilar_cleaned", "DX"),
        ("KronikHastalik_cleaned", "CHR"),
        ("Alerji_cleaned", "ALG"),
        ("UygulamaYerleri_grouped", "SITE"),
        ("TedaviAdi_cleaned", "TX"),
    ]:
        if src in df.columns:
            df[f"{pfx}__missing"] = df[src].isna().astype("Int8")

    # Multi-hot kolonlar + rapor için ayrıntıları topla
    multihot_info: List[dict] = []

    if "Tanilar_cleaned" in df.columns:
        df, _ = add_multi_hot(df, "Tanilar_cleaned", prefix="DX", top_n=30)
        multihot_info.append({"source": "Tanilar_cleaned", "prefix": "DX__", "rule": "top_n=30", "other": True})

    if "KronikHastalik_cleaned" in df.columns:
        df, _ = add_multi_hot(df, "KronikHastalik_cleaned", prefix="CHR", top_n=20)
        multihot_info.append({"source": "KronikHastalik_cleaned", "prefix": "CHR__", "rule": "top_n=20", "other": True})

    if "Alerji_cleaned" in df.columns:
        df, _ = add_multi_hot(df, "Alerji_cleaned", prefix="ALG", top_n=10)
        multihot_info.append({"source": "Alerji_cleaned", "prefix": "ALG__", "rule": "top_n=10", "other": True})

    if "UygulamaYerleri_grouped" in df.columns:
        df, _ = add_multi_hot(df, "UygulamaYerleri_grouped", prefix="SITE", top_n=None, make_other=False)
        multihot_info.append({"source": "UygulamaYerleri_grouped", "prefix": "SITE__", "rule": "tümü (top_n=None)", "other": False})

    if "TedaviAdi_cleaned" in df.columns:
        df, _ = add_multi_hot(df, "TedaviAdi_cleaned", prefix="TX", top_n=15)
        multihot_info.append({"source": "TedaviAdi_cleaned", "prefix": "TX__", "rule": "top_n=15", "other": True})

    # ---- MODEL HAZIR TABLO ----
    model_ready, report = build_model_ready(df)

    # ---- KONSOL RAPORU ----
    print_processing_report(cat_cols, report, multihot_info)

    # ---- KAYITLAR ----
    rows_path = os.path.join("data", "rows.xlsx")
    df.to_excel(rows_path, index=False)
    print(f">>> Satır verisi kaydedildi: {rows_path}")

    bytx_path = os.path.join("data", "by_treatment.xlsx")
    by_tx.to_excel(bytx_path, index=False)
    print(f">>> Tedavi bazlı özet kaydedildi: {bytx_path}")

    model_ready_path = os.path.join("data", "model_ready.xlsx")
    model_ready.to_excel(model_ready_path, index=False)
    print(f">>> Modellemeye hazır veri kaydedildi: {model_ready_path}")
