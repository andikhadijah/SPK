# app.py (FINAL - Improved)
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
import os
import re

# ---------- Config ----------
APP_DB = 'spk_results.db'
# prefer absolute path to uploaded excel (as you uploaded)
EXCEL_PATH = '/mnt/data/data_hasil_gabungan.xlsx'
EXCEL_FILES = [EXCEL_PATH, 'data_hasil_gabungan.xlsx', 'data_hasil_gabungan.xls']
DEFAULT_ALPHA = 0.5
WEIGHT_SUM_TOL = 0.001

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-this'  # change in production
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{APP_DB}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------- DB Model ----------
class Production(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    produksi = db.Column(db.String(256))
    provinsi = db.Column(db.String(128))
    volume = db.Column(db.Float)
    nilai = db.Column(db.Float)
    harga = db.Column(db.Float)
    saw_score = db.Column(db.Float)
    topsis_score = db.Column(db.Float)
    composite = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ---------- Utilities ----------
def find_excel_file():
    for name in EXCEL_FILES:
        if name and os.path.exists(name):
            return name
    return None

def to_numeric_safe(x):
    """
    Convert many possible string number formats to float.
    Handles Indonesian formatting (1.234,56) and plain numbers.
    Returns None if cannot parse.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == '':
        return None
    # remove non-breaking spaces
    s = s.replace('\xa0', '')
    # if there are currency symbols or text — remove them
    s = re.sub(r'[^\d,.\-]', '', s)
    # If both dot and comma present, assume dot thousands comma decimal (ID style) OR dot decimal?
    # Heuristic:
    if s.count('.') > 0 and s.count(',') > 0:
        # assume dot thousands, comma decimal -> remove dots, replace comma with dot
        s = s.replace('.', '').replace(',', '.')
    else:
        # only commas -> if more than one comma, remove thousands separators (commas)
        if s.count(',') > 1 and s.count('.') == 0:
            s = s.replace(',', '')
        # single comma and no dot -> comma as decimal separator
        elif s.count(',') == 1 and s.count('.') == 0:
            s = s.replace(',', '.')
        # multiple dots and no comma -> remove thousands dots
        elif s.count('.') > 1 and s.count(',') == 0:
            s = s.replace('.', '')
    try:
        return float(s)
    except:
        # fallback: strip anything non-digit except dot/minus
        cleaned = re.sub(r'[^\d.\-]', '', s)
        try:
            return float(cleaned) if cleaned != '' else None
        except:
            return None

def clean_prov_name(s):
    if pd.isna(s):
        return ''
    return str(s).strip()

def load_and_prepare_dataframe():
    """
    Read Excel using fixed column names (exact mapping), parse numeric columns,
    return dataframe with standardized columns:
    ['produksi','provinsi','volume','nilai','harga']
    """
    fname = find_excel_file()
    if not fname:
        raise FileNotFoundError(f"File Excel {EXCEL_PATH} tidak ditemukan di folder proyek.")
    # read with pandas (let pandas infer header)
    df = pd.read_excel(fname, engine='openpyxl')
    # show original columns for debugging (comment out in production)
    # print("Excel columns:", list(df.columns))

    # expected headers in your Excel (exact)
    expected_mapping = {
        'Produksi Ikan Hias': 'produksi',
        'Provinsi': 'provinsi',
        'Volume (Ekor)': 'volume',
        'Nilai (Rp. Juta)': 'nilai',
        'Harga Rata-Rata Tertimbang (Rp/ekor)': 'harga'
    }

    # Normalize column names (strip)
    df.rename(columns={c: c.strip() if isinstance(c, str) else c for c in df.columns}, inplace=True)

    # Check all required columns exist
    missing = [k for k in expected_mapping.keys() if k not in df.columns]
    if missing:
        raise ValueError(f"Kolom yang dibutuhkan tidak ditemukan di Excel: {missing}. "
                         "Pastikan header Excel persis: " + ", ".join(expected_mapping.keys()))

    # rename to standard keys
    df = df.rename(columns=expected_mapping)

    # clean provinsi and produksi strings
    df['produksi'] = df['produksi'].astype(str).str.strip()
    df['provinsi'] = df['provinsi'].astype(str).str.strip()

    # parse numeric columns safely
    df['volume'] = df['volume'].apply(to_numeric_safe)
    df['nilai'] = df['nilai'].apply(to_numeric_safe)
    df['harga'] = df['harga'].apply(to_numeric_safe)

    # drop rows with missing numeric essential criteria
    df = df.dropna(subset=['volume', 'nilai', 'harga']).reset_index(drop=True)

    # ensure proper types
    df['volume'] = df['volume'].astype(float)
    df['nilai'] = df['nilai'].astype(float)
    df['harga'] = df['harga'].astype(float)

    # standardize provinsi case (store original too if needed)
    df['provinsi_clean'] = df['provinsi'].str.upper().str.strip()

    return df

# ---------- Calculation helpers ----------
def compute_saw_basic(vals, weights):
    maxs = vals.max(axis=0)
    maxs[maxs == 0] = 1.0
    saw_norm = vals / maxs
    saw_scores = (saw_norm * weights).sum(axis=1)
    return saw_norm, saw_scores

def compute_topsis_basic(vals, weights):
    denom = np.sqrt((vals**2).sum(axis=0))
    denom[denom == 0] = 1.0
    r = vals / denom
    v = r * weights
    v_pos = v.max(axis=0)
    v_neg = v.min(axis=0)
    d_pos = np.sqrt(((v - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - v_neg) ** 2).sum(axis=1))
    topsis_scores = d_neg / (d_pos + d_neg + 1e-12)
    return r, v, v_pos, v_neg, d_pos, d_neg, topsis_scores

def compute_matrices_for_df(df, weights, alpha=DEFAULT_ALPHA):
    cols = ['volume', 'nilai', 'harga']
    vals = df[cols].values.astype(float)
    w = np.array(weights).astype(float)
    saw_norm, saw_scores = compute_saw_basic(vals, w)
    r, v, v_pos, v_neg, d_pos, d_neg, topsis_scores = compute_topsis_basic(vals, w)
    composite = alpha * saw_scores + (1.0 - alpha) * topsis_scores

    return {
        'index': df['produksi'].tolist(),
        'provinsi': df['provinsi'].tolist(),
        'headers': cols,
        'saw_norm': saw_norm.tolist(),
        'saw_scores': saw_scores.tolist(),
        'topsis_r': r.tolist(),
        'topsis_v': v.tolist(),
        'v_pos': v_pos.tolist(),
        'v_neg': v_neg.tolist(),
        'd_pos': d_pos.tolist(),
        'd_neg': d_neg.tolist(),
        'topsis_scores': topsis_scores.tolist(),
        'composite': composite.tolist()
    }

# ---------- Compute & persist ----------
def compute_and_persist(weights=None, alpha=DEFAULT_ALPHA):
    """
    Compute SAW/TOPSIS for entire Excel dataset and persist to DB.
    This writes full dataset to DB so DB == Excel content + computed scores.
    """
    if weights is None:
        weights = [0.4, 0.4, 0.2]
    df = load_and_prepare_dataframe()
    vals = df[['volume', 'nilai', 'harga']].values.astype(float)

    # compute
    saw_norm, saw_scores = compute_saw_basic(vals, np.array(weights))
    topsis_r, topsis_v, v_pos, v_neg, d_pos, d_neg, topsis_scores = compute_topsis_basic(vals, np.array(weights))
    composite = alpha * saw_scores + (1.0 - alpha) * topsis_scores

    # persist (clear previous)
    Production.query.delete()
    db.session.commit()

    now = datetime.utcnow()
    rows = []
    for i, row in df.iterrows():
        p = Production(
            produksi = row['produksi'],
            provinsi = row['provinsi'],
            volume = float(row['volume']),
            nilai = float(row['nilai']),
            harga = float(row['harga']),
            saw_score = float(saw_scores[i]),
            topsis_score = float(topsis_scores[i]),
            composite = float(composite[i]),
            created_at = now
        )
        rows.append(p)
    if rows:
        db.session.bulk_save_objects(rows)
        db.session.commit()
    return len(rows)

def compute_and_persist_custom(weights, alpha=DEFAULT_ALPHA):
    weights = [float(w) for w in weights]
    return compute_and_persist(weights=weights, alpha=alpha)

# ---------- Routes ----------
@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    uploaded_template = EXCEL_PATH  # show the local path of the uploaded excel
    return render_template('dashboard.html', uploaded_template=uploaded_template)

@app.route('/index')
def index():
    try:
        df = load_and_prepare_dataframe()
        # unique provinsi cleaned but show original names in dropdown
        prov_list = sorted(df['provinsi'].unique().tolist())
    except Exception as e:
        prov_list = []
        flash(str(e), 'error')
    weights = session.get('weights', [0.4, 0.4, 0.2])
    return render_template('index.html', prov_list=prov_list, weights=weights)

@app.route('/compute', methods=['POST'])
def compute():
    prov_selected = request.form.getlist('provinsi')
    prov_selected = [p.strip() for p in prov_selected if p.strip() != '']
    if len(prov_selected) == 0:
        flash('Silakan pilih minimal satu provinsi (atau Pilih Semua).', 'error')
        return redirect(url_for('index'))

    # validate weights
    try:
        w1 = float(request.form.get('w1', '0.0'))
        w2 = float(request.form.get('w2', '0.0'))
        w3 = float(request.form.get('w3', '0.0'))
    except ValueError:
        flash('Bobot tidak valid. Pastikan Anda memasukkan angka desimal, mis. 0.4', 'error')
        return redirect(url_for('index'))

    total = w1 + w2 + w3
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        flash(f'Bobot harus berjumlah 1. Saat ini jumlah = {total:.4f}. Silakan sesuaikan sehingga total = 1.', 'error')
        return redirect(url_for('index'))

    # normalize tiny rounding
    w1, w2, w3 = w1/total, w2/total, w3/total

    session['provinsi'] = prov_selected
    session['weights'] = [w1, w2, w3]

    try:
        # compute & persist full dataset
        compute_and_persist_custom(weights=[w1, w2, w3])
        flash('Perhitungan SAW+TOPSIS selesai dan tersimpan. Silakan lanjutkan ke penjelasan.', 'success')
    except Exception as e:
        flash(f'Gagal menghitung: {e}', 'error')
        return redirect(url_for('index'))

    return redirect(url_for('proses'))

@app.route('/proses')
def proses():
    # load and filter by exact provinsi matches (case-insensitive)
    df = load_and_prepare_dataframe()
    provs = session.get('provinsi', [])
    if provs and len(provs) > 0:
        # match exact after stripping and uppercasing
        provs_clean = [p.upper().strip() for p in provs]
        df_sel = df[df['provinsi'].str.upper().str.strip().isin(provs_clean)].reset_index(drop=True)
    else:
        df_sel = df.copy()

    if df_sel.shape[0] == 0:
        flash('Tidak ada data untuk provinsi yang dipilih.', 'error')
        return redirect(url_for('index'))

    weights = session.get('weights', [0.4, 0.4, 0.2])
    matrices = compute_matrices_for_df(df_sel, weights, alpha=DEFAULT_ALPHA)

    formulas = {
        'saw_norm': 'r_{ij} = x_{ij} / max_j(x_{ij})',
        'saw_score': 'S_i = Σ_j (w_j * r_{ij})',
        'topsis_norm': 'r_{ij} = x_{ij} / sqrt(Σ_i x_{ij}^2)',
        'topsis_weighted': 'v_{ij} = w_j * r_{ij}',
        'topsis_distance': "D_i^+ = sqrt(Σ_j (v_{ij} - v_j^+)^2),  D_i^- = sqrt(Σ_j (v_{ij} - v_j^-)^2)",
        'topsis_score': 'C_i = D_i^- / (D_i^+ + D_i^-)',
        'hybrid': 'Composite = α * SAW + (1-α) * TOPSIS'
    }

    return render_template('proses.html',
                           matrices=matrices,
                           formulas=formulas,
                           weights=weights,
                           provs=provs)

@app.route('/results')
def results():
    provs = session.get('provinsi', None)
    # ensure DB populated
    if Production.query.count() == 0:
        saved_weights = session.get('weights', [0.4, 0.4, 0.2])
        compute_and_persist_custom(weights=saved_weights)

    q = Production.query
    if provs and len(provs) > 0:
        # exact match (case-insensitive)
        from sqlalchemy import or_
        filters = [Production.provinsi.ilike(p.strip()) for p in provs]
        # We want exact matching ignoring case; SQLAlchemy ilike with full equality
        # build filters for equality in upper-case (safer)
        # fallback: use ilike with exact string
        q = q.filter(or_(*filters))
        current_label = ', '.join(provs)
    else:
        current_label = 'SEMUA'

    items = q.order_by(Production.composite.desc()).all()
    ready_to_show = True
    return render_template('results.html', items=items, provinsi=current_label, ready=ready_to_show)

@app.route('/recompute', methods=['POST'])
def recompute():
    weights = session.get('weights', [0.4,0.4,0.2])
    compute_and_persist_custom(weights=weights)
    flash('Recompute selesai dengan bobot saat ini.', 'success')
    return redirect(url_for('results'))

# small utility route to download the source Excel (optional)
@app.route('/download-excel')
def download_excel():
    fname = find_excel_file()
    if not fname:
        flash('File Excel tidak ditemukan.', 'error')
        return redirect(url_for('dashboard'))
    # direct link path for convenience — served by your webserver or environment
    return redirect(f'file://{os.path.abspath(fname)}')

# ---------- Entry ----------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
