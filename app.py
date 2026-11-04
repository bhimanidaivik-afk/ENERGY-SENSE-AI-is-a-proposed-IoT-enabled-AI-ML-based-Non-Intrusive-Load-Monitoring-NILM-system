import sys
import json
sys.path.append("src")

import numpy as np
import pandas as pd
import torch
import streamlit as st
from model import Seq2Point

CSV = "data/house1_ukdale.csv"
WINDOW, HALF = 99, 49
THRESH = {"fridge": 50, "kettle": 1000, "microwave": 200}
APPLIANCES = list(THRESH)
ICON = {"fridge": "🧊", "kettle": "☕", "microwave": "🍲"}

st.set_page_config(page_title="NILM Energy Dashboard", page_icon="⚡", layout="wide")


@st.cache_resource
def load_model(name):
    m = Seq2Point()
    m.load_state_dict(torch.load(f"models/{name}.pt"))
    m.eval()
    return m


def predict(model, agg, s):
    x = ((agg - s["x_mean"]) / s["x_std"]).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(x, WINDOW).copy()
    t = torch.tensor(windows).unsqueeze(1)
    out = []
    with torch.no_grad():
        for i in range(0, len(t), 1024):
            out.append(model(t[i:i + 1024]).numpy())
    pred = np.clip(np.concatenate(out) * s["y_std"] + s["y_mean"], 0, None)
    full = np.zeros(len(agg))
    full[HALF:HALF + len(pred)] = pred
    return full


@st.cache_data
def load_results():
    df = pd.read_csv(CSV, parse_dates=["timestamp"])
    n_train = int(0.8 * (len(df) - WINDOW))
    test = df.iloc[n_train:].reset_index(drop=True)   # sirf wo data jo model ne kabhi nahi dekha
    agg = test["aggregate"].values.astype(np.float32)
    stats = json.load(open("models/stats.json"))
    res = pd.DataFrame({"timestamp": test["timestamp"], "aggregate": test["aggregate"]})
    for n in APPLIANCES:
        res[n] = predict(load_model(n), agg, stats[n])
        res[n + "_actual"] = test[n]
    return res.iloc[HALF:len(res) - HALF].reset_index(drop=True)


def events(x, thr):
    on = x > thr
    return int((on[1:] & ~on[:-1]).sum())


res = load_results()

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
rate = st.sidebar.number_input("Bijli ka rate (₹ per unit)", value=7.0, step=0.5)
dmin, dmax = res["timestamp"].dt.date.min(), res["timestamp"].dt.date.max()
rng = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(rng, (tuple, list)) and len(rng) == 2:
    d0, d1 = rng
elif isinstance(rng, (tuple, list)) and len(rng) == 1:
    d0 = d1 = rng[0]
else:
    d0 = d1 = rng
st.sidebar.caption("Dashboard sirf test data dikhata hai, jo model ki training mein use nahi hua.")

mask = (res["timestamp"].dt.date >= d0) & (res["timestamp"].dt.date <= d1)
view = res[mask]
if view.empty:
    st.warning("Is date range mein data nahi hai.")
    st.stop()

days = max(len(view) / 1440, 1 / 1440)
kwh = {n: view[n].sum() / 60 / 1000 for n in APPLIANCES}
kwh_actual = {n: view[n + "_actual"].sum() / 60 / 1000 for n in APPLIANCES}
total_kwh = view["aggregate"].sum() / 60 / 1000
other = max(total_kwh - sum(kwh.values()), 0)


def month(k):
    return k / days * 30


# ---------- Header + cards ----------
st.title("⚡ Smart NILM: Ek Meter, Har Appliance ka Hisaab")
st.caption(f"UK-DALE House 1 | {d0} se {d1} | AI model ghar ke total power se appliance-wise bijli nikalta hai")

c = st.columns(4)
c[0].metric("🏠 Ghar ka total", f"{total_kwh:.1f} kWh", f"₹ {total_kwh * rate:.0f}", delta_color="off")
for col, n in zip(c[1:], APPLIANCES):
    col.metric(f"{ICON[n]} {n.capitalize()}", f"{kwh[n]:.2f} kWh", f"₹ {kwh[n] * rate:.0f}", delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Appliance Detail", "🎯 Model Accuracy", "💡 Bachat Tips"])

# ---------- Tab 1 ----------
with tab1:
    left, right = st.columns(2)
    split = {n.capitalize(): kwh[n] for n in APPLIANCES}
    split["Baaki appliances"] = other
    with left:
        st.subheader("Bijli kahan gayi? (kWh)")
        st.bar_chart(pd.Series(split))
    with right:
        st.subheader("Kharcha (₹)")
        table = pd.DataFrame({
            "kWh": {k: round(v, 2) for k, v in split.items()},
            "₹": {k: round(v * rate, 1) for k, v in split.items()},
            "Share %": {k: round(v / max(total_kwh, 1e-9) * 100, 1) for k, v in split.items()},
        })
        st.dataframe(table)
        st.write(f"Teeno tracked appliances ka share: **{sum(kwh.values()) / max(total_kwh, 1e-9) * 100:.1f}%**")

    st.subheader("Ghar ka total power (W)")
    st.line_chart(view.set_index("timestamp")["aggregate"])
    st.subheader("Appliance-wise power (AI ne alag kiya)")
    st.area_chart(view.set_index("timestamp")[APPLIANCES])

# ---------- Tab 2 ----------
with tab2:
    name = st.selectbox("Appliance chuno", APPLIANCES)
    day_list = sorted(view["timestamp"].dt.date.unique())
    day = st.selectbox("Din chuno", day_list)
    dv = view[view["timestamp"].dt.date == day]

    a, b, c2 = st.columns(3)
    a.metric("Predicted energy", f"{kwh[name]:.2f} kWh")
    b.metric("Actual energy", f"{kwh_actual[name]:.2f} kWh")
    a2 = events(view[name + "_actual"].values, THRESH[name])
    p2 = events(view[name].values, THRESH[name])
    c2.metric("ON hone ki baar (Predicted / Actual)", f"{p2} / {a2}")

    st.subheader(f"{name.capitalize()}: Predicted vs Actual ({day})")
    st.line_chart(pd.DataFrame({
        "Actual": dv[name + "_actual"].values,
        "Predicted": dv[name].values,
    }, index=dv["timestamp"]))

# ---------- Tab 3 ----------
with tab3:
    st.subheader("Model kitna sahi hai?")
    try:
        m = pd.DataFrame(json.load(open("models/metrics.json"))).T
        st.dataframe(m)
    except FileNotFoundError:
        st.warning("models/metrics.json nahi mili. Pehle python src/train.py chalao.")
    st.markdown(
        "- **MAE (W)**: average kitne watt ka galat andaza (kam = achha)\n"
        "- **F1**: appliance ON/OFF sahi pehchanne ka score (1 ke paas = achha)\n"
        "- **Energy error (%)**: kul bijli ke andaze mein kitna farak (kam = achha)\n\n"
        "Ye numbers us data par hain jo model ne training mein kabhi nahi dekha."
    )

# ---------- Tab 4 ----------
with tab4:
    st.subheader("Mahine ka andaza")
    for n in APPLIANCES:
        st.write(f"{ICON[n]} **{n.capitalize()}**: {month(kwh[n]):.1f} kWh, lagbhag ₹ {month(kwh[n]) * rate:.0f} per mahina")

    st.subheader("Agar use kam karein to?")
    cut = st.slider("Kettle aur microwave ka use kitna kam karoge (%)", 0, 50, 20)
    saving = (month(kwh["kettle"]) + month(kwh["microwave"])) * cut / 100 * rate
    st.success(f"Mahine ki bachat: lagbhag ₹ {saving:.0f}")

    st.subheader("Tips")
    top = max(kwh, key=kwh.get)
    st.write(f"Sabse zyada bijli **{top}** use kar raha hai, pehle uspe dhyan do.")
    st.markdown(
        "- 🧊 Fridge: door ka seal check karo, garam cheez andar mat rakho, peeche hawa ke liye jagah chhodo\n"
        "- ☕ Kettle: sirf zaroorat jitna paani ubalo\n"
        "- 🍲 Microwave: chhoti matra ke liye alag baar-baar chalane se bacho"
    )