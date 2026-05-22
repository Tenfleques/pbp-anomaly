"""
Data parsers for common environmental sensor datasets.

Loaders return a standardized dict:
    {
        'name': str,
        'data': pd.DataFrame,
        'sensor_cols': list[str],
        'ref_cols': list[str],
    }
"""

import os
import numpy as np
import pandas as pd


# ─── NOAA ISD (weather stations) ────────────────────────────────────────

def load_noaa_isd(data_dir, prefix='ord', years=(2022, 2023), station_name='NOAA ISD'):
    """Load and parse NOAA ISD hourly METAR observations.

    Args:
        data_dir: directory containing {prefix}_{year}.csv files
        prefix: file prefix (e.g. 'ord', 'mia', 'sfo', 'fai')
        years: tuple of years to load
        station_name: display name

    Returns:
        dict with name, data, sensor_cols, ref_cols
    """
    dfs = []
    for year in years:
        path = os.path.join(data_dir, f'{prefix}_{year}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        dfs.append(pd.read_csv(path, low_memory=False))

    df = pd.concat(dfs, ignore_index=True)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df[df['REPORT_TYPE'] == 'FM-15'].copy()
    df['hour'] = df['DATE'].dt.floor('h')
    df = df.sort_values('DATE').groupby('hour').first().reset_index()
    df['DATE'] = df['hour']

    parsed = pd.DataFrame({'DATE': df['DATE']})
    parsed['Temp'] = _parse_signed(df['TMP'])
    parsed['DewP'] = _parse_signed(df['DEW'])
    parsed['Pres'] = _parse_pressure(df['SLP'], df.get('MA1'))
    parsed['Wind'] = _parse_wind(df['WND'])
    parsed['Vis'] = _parse_unsigned(df['VIS'], missing=999999)
    parsed['Ceil'] = _parse_unsigned(df['CIG'], missing=99999)

    sensor_cols = ['Temp', 'DewP', 'Pres', 'Wind', 'Vis', 'Ceil']
    parsed = parsed.dropna(subset=sensor_cols).reset_index(drop=True)

    return {
        'name': station_name,
        'data': parsed,
        'sensor_cols': sensor_cols,
        'ref_cols': sensor_cols,
    }


def _parse_signed(series, scale=10.0, missing=9999):
    r = []
    for val in series:
        try:
            raw = str(val).split(',')[0]
            num = float(raw) / scale
            r.append(np.nan if abs(num * scale) >= missing else num)
        except (ValueError, IndexError):
            r.append(np.nan)
    return pd.Series(r, index=series.index)


def _parse_unsigned(series, missing=999999):
    r = []
    for val in series:
        try:
            num = float(str(val).split(',')[0])
            r.append(np.nan if num >= missing else num)
        except (ValueError, IndexError):
            r.append(np.nan)
    return pd.Series(r, index=series.index)


def _parse_wind(series):
    r = []
    for val in series:
        try:
            spd = float(str(val).split(',')[3]) / 10.0
            r.append(np.nan if spd >= 99.9 else spd)
        except (ValueError, IndexError):
            r.append(np.nan)
    return pd.Series(r, index=series.index)


def _parse_pressure(slp, ma1=None):
    r = []
    ma1_vals = ma1.values if ma1 is not None else [None] * len(slp)
    for s, m in zip(slp, ma1_vals):
        try:
            raw = int(str(s).split(',')[0])
            if raw != 99999:
                r.append(raw / 10.0)
                continue
        except (ValueError, IndexError):
            pass
        try:
            raw = int(str(m).split(',')[0])
            if raw != 99999:
                r.append(raw / 10.0)
                continue
        except (ValueError, IndexError):
            pass
        r.append(np.nan)
    return pd.Series(r, index=slp.index)


# ─── UCI Air Quality ────────────────────────────────────────────────────

def load_uci_air_quality(csv_path):
    """Load UCI Air Quality dataset.

    Args:
        csv_path: path to AirQualityUCI.csv

    Returns:
        dict with name, data, sensor_cols, ref_cols
    """
    df = pd.read_csv(csv_path, sep=';', decimal=',', na_values=-200)
    df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)

    sensor_cols = [c for c in df.columns if 'PT08' in c or c in ['T', 'RH', 'AH']]
    ref_cols = [c for c in df.columns if c in ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']]

    data = df[sensor_cols + ref_cols].dropna()
    return {
        'name': 'UCI Air Quality (Italy)',
        'data': data.reset_index(drop=True),
        'sensor_cols': sensor_cols,
        'ref_cols': ref_cols,
    }


# ─── Beijing Multi-Site ─────────────────────────────────────────────────

def load_beijing_multisite(data_dir, station='Dongsi'):
    """Load Beijing Multi-Site Air Quality dataset.

    Args:
        data_dir: directory containing PRSA_Data_*.csv files
        station: station name to use (default: Dongsi)

    Returns:
        dict with name, data, sensor_cols, ref_cols
    """
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    # Find target station
    target_file = None
    for path in sorted(csv_files):
        if station in os.path.basename(path):
            target_file = path
            break
    if target_file is None:
        target_file = sorted(csv_files)[0]

    df = pd.read_csv(target_file)
    pollutant_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    meteo_cols = ['TEMP', 'PRES', 'DEWP', 'WSPM']
    sensor_cols = [c for c in pollutant_cols + meteo_cols if c in df.columns]
    ref_cols = [c for c in pollutant_cols if c in df.columns]

    df = df.dropna(subset=[c for c in sensor_cols if c in df.columns])
    data = df[sensor_cols].copy()

    station_name = os.path.basename(target_file).replace('PRSA_Data_', '').split('_')[0]
    return {
        'name': f'Beijing Multi-Site ({station_name})',
        'data': data.reset_index(drop=True),
        'sensor_cols': sensor_cols,
        'ref_cols': ref_cols,
    }


# ─── Open-Meteo CAMS (São Paulo, Cape Town, etc.) ──────────────────────

def load_open_meteo_csv(csv_path, city_name='City'):
    """Load Open-Meteo CAMS air quality CSV.

    Args:
        csv_path: path to air_quality.csv (pre-downloaded from Open-Meteo API)
        city_name: display name

    Returns:
        dict with name, data, sensor_cols, ref_cols
    """
    df = pd.read_csv(csv_path, skiprows=2)

    col_rename = {
        'pm10': 'PM10', 'pm2_5': 'PM2.5',
        'carbon_monoxide': 'CO', 'nitrogen_dioxide': 'NO2',
        'sulphur_dioxide': 'SO2', 'ozone': 'O3',
    }
    for old, new in list(col_rename.items()):
        for c in df.columns:
            if old in c.lower():
                col_rename[c] = new
                break
    df = df.rename(columns=col_rename)

    sensor_cols = [c for c in ['PM10', 'PM2.5', 'CO', 'NO2', 'SO2', 'O3'] if c in df.columns]
    df = df[sensor_cols].apply(pd.to_numeric, errors='coerce').dropna()

    return {
        'name': f'{city_name} (CAMS)',
        'data': df.reset_index(drop=True),
        'sensor_cols': sensor_cols,
        'ref_cols': sensor_cols,
    }


# ─── Generic CSV loader ────────────────────────────────────────────────

# ─── EPA AQS (US air quality) ──────────────────────────────────────────

def load_epa_aqs(data_dir, year=2023, site_state='06', site_county='037', site_num='1103'):
    """Load US EPA AQS hourly data for a single site by streaming from ZIP files.

    Default: Los Angeles site 06-037-1103 (multi-pollutant urban).
    Reads directly from ZIP using chunked iteration to avoid loading 2GB CSVs into RAM.

    Pollutant parameter codes:
      44201=O3, 42401=SO2, 42101=CO, 42602=NO2, 88101=PM2.5

    Args:
        data_dir: directory to store/find downloaded ZIPs
        year: data year
        site_state: 2-digit FIPS state code
        site_county: 3-digit county code
        site_num: 4-digit site number

    Returns:
        dict with name, data, sensor_cols, ref_cols
    """
    import zipfile
    import urllib.request

    epa_dir = os.path.join(data_dir, f'epa_aqs_{year}')
    os.makedirs(epa_dir, exist_ok=True)

    params = {
        'O3': '44201',
        'SO2': '42401',
        'CO': '42101',
        'NO2': '42602',
        'PM2.5': '88101',
    }

    dfs = {}
    for name, code in params.items():
        zip_path = os.path.join(epa_dir, f'hourly_{code}_{year}.zip')

        if not os.path.exists(zip_path):
            url = f"https://aqs.epa.gov/aqsweb/airdata/hourly_{code}_{year}.zip"
            print(f"  Downloading EPA {name} ({code})...")
            try:
                urllib.request.urlretrieve(url, zip_path)
            except Exception as e:
                print(f"  WARNING: Could not download {name}: {e}")
                continue

        if not os.path.exists(zip_path):
            continue

        try:
            with zipfile.ZipFile(zip_path) as zf:
                csv_name = None
                for member in zf.namelist():
                    if member.endswith('.csv'):
                        csv_name = member
                        break
                if csv_name is None:
                    print(f"  WARNING: No CSV found in {zip_path}")
                    continue

                print(f"  Reading EPA {name} from ZIP (filtering site "
                      f"{site_state}-{site_county}-{site_num})...")
                site_rows = []
                chunks = pd.read_csv(
                    zf.open(csv_name),
                    chunksize=50000,
                    dtype={'State Code': str, 'County Code': str, 'Site Num': str},
                    low_memory=False,
                )
                for chunk in chunks:
                    filtered = chunk[
                        (chunk['State Code'] == site_state) &
                        (chunk['County Code'] == site_county) &
                        (chunk['Site Num'] == site_num)
                    ]
                    if len(filtered) > 0:
                        site_rows.append(filtered)

                if not site_rows:
                    print(f"  EPA {name}: site not found in data")
                    continue

                site_df = pd.concat(site_rows, ignore_index=True)
                site_df['datetime'] = pd.to_datetime(
                    site_df['Date Local'] + ' ' + site_df['Time Local']
                )
                site_df = site_df[['datetime', 'Sample Measurement']].rename(
                    columns={'Sample Measurement': name}
                )
                site_df = site_df.groupby('datetime').mean().reset_index()
                dfs[name] = site_df
                print(f"  EPA {name}: {len(site_df)} hourly readings")

        except Exception as e:
            print(f"  WARNING: Error reading {name} from ZIP: {e}")
            continue

    if len(dfs) < 3:
        raise ValueError(f"Only {len(dfs)} pollutants found for site. Need at least 3.")

    merged = None
    for name, df in dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on='datetime', how='inner')

    merged = merged.sort_values('datetime').reset_index(drop=True)
    merged = merged.dropna()

    sensor_cols = [c for c in merged.columns if c != 'datetime']
    ref_cols = sensor_cols.copy()

    return {
        'name': f'EPA AQS (LA, {year})',
        'data': merged[sensor_cols],
        'sensor_cols': sensor_cols,
        'ref_cols': ref_cols,
    }


# ─── Generic CSV loader ────────────────────────────────────────────────

def load_csv(csv_path, sensor_cols, ref_cols=None, name='Custom'):
    """Load any CSV with specified sensor columns.

    Args:
        csv_path: path to CSV file
        sensor_cols: list of column names to use as sensors
        ref_cols: columns for ground truth (default: same as sensor_cols)
        name: dataset display name

    Returns:
        dict with name, data, sensor_cols, ref_cols
    """
    df = pd.read_csv(csv_path)
    if ref_cols is None:
        ref_cols = sensor_cols

    missing = [c for c in sensor_cols + ref_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    all_cols = list(set(sensor_cols + ref_cols))
    df = df[all_cols].apply(pd.to_numeric, errors='coerce').dropna()

    return {
        'name': name,
        'data': df.reset_index(drop=True),
        'sensor_cols': sensor_cols,
        'ref_cols': ref_cols,
    }
