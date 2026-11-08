"""StressGuard AI - versi Streamlit.

Jalankan:  py -m streamlit run app.py
Model Random Forest dibaca dari stress_model.json (hasil export_model_json.py).
"""
import streamlit as st

from stress_engine import load_meta, load_model, predict_proba

st.set_page_config(page_title="StressGuard AI", page_icon="🌱", layout="wide")

# ----------------------------------------------------------------------------
# Definisi input (urutan & nama fitur harus sama dengan model_meta.json)
# kind: "float" / "int" / "percent" (attendance_rate disimpan 0-1, tampil 0-100%)
# ----------------------------------------------------------------------------
GROUPS = {
    "Kebiasaan harian": [
        dict(name="sleep_hours_per_day", label="Jam tidur per hari", caption="Rata-rata tidurmu dalam sehari.",
            min=4.0, max=10.0, step=0.5, default=7.0, kind="float", unit="jam"),
        dict(name="study_hours_per_week", label="Jam belajar per minggu", caption="Total waktu belajar di luar kelas.",
            min=2, max=35, step=1, default=15, kind="int", unit="jam"),
    ],
    "Beban akademik": [
        dict(name="academic_pressure_score", label="Tekanan akademik", caption="0 = santai, 10 = sangat tertekan.",
            min=0.0, max=10.0, step=0.5, default=5.0, kind="float", unit=""),
        dict(name="attendance_rate", label="Kehadiran kuliah", caption="Persentase kelas yang kamu hadiri.",
            min=0, max=100, step=1, default=85, kind="percent", unit="%"),
        dict(name="missed_submissions_count", label="Tugas terlewat", caption="Jumlah tugas yang telat/tidak dikumpulkan.",
            min=0, max=20, step=1, default=2, kind="int", unit="tugas"),
    ],
    "Kondisi emosional": [
        dict(name="anxiety_score", label="Kecemasan", caption="0 = tenang, 10 = sangat cemas.",
            min=0.0, max=10.0, step=0.5, default=5.0, kind="float", unit=""),
        dict(name="depression_score", label="Suasana hati rendah", caption="0 = ceria, 10 = sangat murung.",
            min=0.0, max=10.0, step=0.5, default=5.0, kind="float", unit=""),
        dict(name="burnout_score", label="Kelelahan (burnout)", caption="0 = segar, 10 = sangat lelah.",
            min=0.0, max=10.0, step=0.5, default=5.0, kind="float", unit=""),
    ],
}
ALL_FIELDS = [f for fields in GROUPS.values() for f in fields]

RESULTS = {
    "Rendah": dict(
        emoji="🌿", color="#2e9e6b", title="Stres rendah",
        text="Catatan harianmu menunjukkan pola yang cukup seimbang. Pertahankan kebiasaan baik yang sudah berjalan.",
        tips=["Jaga jam tidur yang konsisten.", "Sisihkan waktu untuk hobi atau bertemu teman.",
            "Tetap kenali sinyal tubuhmu sebelum lelah menumpuk."]),
    "Sedang": dict(
        emoji="🌤️", color="#e0a21b", title="Stres sedang",
        text="Ada beberapa tanda tekanan yang mulai terasa. Ini wajar, tapi baik untuk diperhatikan sejak dini.",
        tips=["Pecah tugas besar menjadi langkah kecil.", "Coba tidur lebih teratur dan beri jeda saat belajar.",
            "Ceritakan bebanmu ke teman, keluarga, atau dosen wali."]),
    "Tinggi": dict(
        emoji="🌧️", color="#d9534f", title="Stres tinggi",
        text="Pola harianmu menunjukkan tekanan yang cukup berat. Kamu tidak harus menghadapinya sendirian.",
        tips=["Bicaralah dengan orang yang kamu percaya hari ini.",
            "Hubungi layanan konseling kampus atau psikolog.",
            "Utamakan istirahat dan kurangi beban yang bisa ditunda."]),
}


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_meta():
    return load_meta()


def reset_form():
    for f in ALL_FIELDS:
        st.session_state[f["name"]] = f["default"]


# init state
for f in ALL_FIELDS:
    st.session_state.setdefault(f["name"], f["default"])

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🌱 StressGuard AI")
st.caption("Cek dini tingkat stres mahasiswa dengan machine learning · Kelompok 2 · Universitas Bunda Mulia")

tab_cek, tab_model = st.tabs(["Cek stres", "Tentang model"])

# ----------------------------------------------------------------------------
# Tab 1: form + hasil
# ----------------------------------------------------------------------------
with tab_cek:
    st.info("Sesuaikan slider dengan yang kamu rasakan, bukan yang diharapkan orang lain. "
            "Hasil ini **bukan diagnosis medis**.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Catatan keseharianmu")
        for gname, fields in GROUPS.items():
            with st.container(border=True):
                st.markdown(f"**{gname}**")
                for f in fields:
                    suffix = f" ({f['unit']})" if f["unit"] else ""
                    st.slider(f["label"] + suffix, min_value=f["min"], max_value=f["max"],
                            step=f["step"], key=f["name"], help=f["caption"])
        st.button("Setel ulang", on_click=reset_form)

    # kumpulkan nilai (persen -> 0..1)
    values = {}
    for f in ALL_FIELDS:
        v = st.session_state[f["name"]]
        values[f["name"]] = v / 100 if f["kind"] == "percent" else v

    with right:
        st.subheader("Sebuah gambaran, bukan label.")
        try:
            model = get_model()
            proba = predict_proba(model, values)
            label = max(proba, key=proba.get)
            res = RESULTS[label]

            st.markdown(
                f"""<div style="border-left:8px solid {res['color']};padding:1rem 1.2rem;
                border-radius:8px;background:rgba(128,128,128,0.08)">
                <div style="font-size:2.2rem">{res['emoji']}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{res['color']}">{res['title']}</div>
                <p style="margin-top:.5rem">{res['text']}</p></div>""",
                unsafe_allow_html=True,
            )
            st.markdown("**Langkah kecil yang bisa dicoba:**")
            for tip in res["tips"]:
                st.markdown(f"- {tip}")

            with st.expander("Lihat peluang tiap kategori"):
                for c in model["classes"]:
                    st.write(f"{c}: {proba[c] * 100:.1f}%")
                    st.progress(min(max(proba[c], 0.0), 1.0))
                st.caption("Angka ini hasil rata-rata suara pohon pada Random Forest, "
                        "bukan kepastian.")
        except FileNotFoundError as e:
            st.error(f"{e}\n\nPastikan `stress_model.json` ada di folder yang sama dengan `app.py`.")

# ----------------------------------------------------------------------------
# Tab 2: tentang model
# ----------------------------------------------------------------------------
with tab_model:
    try:
        meta = get_meta()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Bagaimana model ini dibuat")
    c1, c2, c3 = st.columns(3)
    c1.metric("Algoritma", meta["algorithm"])
    c2.metric("Akurasi uji", f"{meta['metrics']['accuracy'] * 100:.2f}%")
    c3.metric("Macro F1", f"{meta['metrics']['macro_f1'] * 100:.2f}%")

    ds = meta["dataset"]
    st.markdown(
        f"- **Dataset:** {ds['name']} oleh {ds['author']} (Kaggle), {ds['rows']:,} baris, {ds['columns']} kolom\n"
        f"- **Validasi:** {meta['validation']}\n"
        f"- **Data uji:** {meta['test_rows']:,} baris\n"
        f"- **Kelas:** {', '.join(meta['classes'])}"
    )

    st.markdown("**Perbandingan algoritma (5-fold CV, macro F1)**")
    st.bar_chart(meta["metrics"]["cv_macro_f1"])

    st.markdown("**Fitur dan rentang nilainya**")
    rows = [{"Fitur": k, "Min": v[0], "Maks": v[1]} for k, v in meta["feature_ranges"].items()]
    st.dataframe(rows, hide_index=True)

    st.warning(
        "**Baca skor dengan hati-hati.** Akurasi ≈ 99,9% biasanya berarti label pada dataset hampir "
        "sepenuhnya ditentukan oleh indikator yang dipakai. Itu **tidak** berarti alat ini 99,9% andal "
        "pada mahasiswa sungguhan."
    )
    st.markdown("**Keterbatasan:** delapan indikator tidak bisa mewakili semua penyebab stres "
                "(keluarga, kesehatan, keuangan, dll.); hasilnya hanya tiga kategori kasar; "
                "tidak ada data yang disimpan.")