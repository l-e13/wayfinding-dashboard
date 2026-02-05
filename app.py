import streamlit as st
import gspread
from gspread_dataframe import get_as_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from scipy.stats import gaussian_kde
from PIL import Image, ImageDraw, UnidentifiedImageError
import json

# Setup Google Sheets API credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = st.secrets["google_sheets"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

# --- Load data (Round 1 + Round 2) ---
@st.cache_data(ttl=3600)
def load_sheet(spreadsheet_name: str, worksheet_name: str) -> pd.DataFrame:
    ws = client.open(spreadsheet_name).worksheet(worksheet_name)
    d = get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all")
    d["id"] = pd.to_numeric(d["id"], errors="coerce").astype("Int64")
    return d

df_r1 = load_sheet("Wayfinding Data", "Raw Data")
df_r2 = load_sheet("Wayfinding Data", "Round 2")

# Merge into one table: one row per participant id
df = df_r1.merge(
    df_r2,
    on="id",
    how="outer",
    suffixes=("", "_rd2dup")
)


ROUND1_IMG = "Floor Plan New.jpg"
ROUND2_IMG = "Round 2 floor plan.png" 


# Page selector
st.set_page_config(layout="wide")

st.title("Wayfinding Performance Study Dashboard")
st.markdown("Use the filters below to compare an individual member to a group average.")
st.markdown("---")



TOTAL_AREA_R1 = 515
TOTAL_AREA_R2 = 512.5

def pick_col(r1_col: str, r2_col: str, is_r2: bool) -> str:
    return r2_col if is_r2 else r1_col

def pick_total_area(is_r2: bool) -> float:
    return TOTAL_AREA_R2 if is_r2 else TOTAL_AREA_R1


def _to_num(x):
    """Coerce values like '482', 482, '482.0' -> float; returns NaN if not parseable."""
    return pd.to_numeric(x, errors="coerce")

def _to_pct01(x):
    """Coerce values like '93.66%', 93.66, 0.9366 -> 0-1 float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip()
        if s.endswith("%"):
            s = s[:-1].strip()
            v = pd.to_numeric(s, errors="coerce")
            return v / 100 if pd.notna(v) else np.nan
        v = pd.to_numeric(s, errors="coerce")
        if pd.isna(v):
            return np.nan
        return v / 100 if v > 1 else float(v)
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return np.nan
    return v / 100 if v > 1 else float(v)

def add_search_derived(df_in: pd.DataFrame, is_r2: bool) -> pd.DataFrame:
    """
    Adds round-aware derived columns used in the Search Performance section:
      - total_ft
      - total_perc01
      - time_called_sec
      - search_rate_sqft_min
      - adjusted_score
      - bed_quadrants_searched (out of 12)
      - percent_beds_searched
    """
    df2 = df_in.copy()

    total_ft_col   = pick_col("total_ft_searched", "rd2_total_ft_searched", is_r2)
    total_perc_col = pick_col("total_perc_searched", "rd2_total_perc_searched", is_r2)
    time_called_col = pick_col("search_called", "rd2_t7_tot_time", is_r2)  # best available Round 2 analog

    df2["total_ft"] = _to_num(df2.get(total_ft_col))
    df2["total_perc01"] = df2.get(total_perc_col).apply(_to_pct01)

    df2["time_called_sec"] = _to_num(df2.get(time_called_col))
    df2["search_rate_sqft_min"] = (df2["total_ft"] / df2["time_called_sec"]) * 60
    df2["adjusted_score"] = df2["search_rate_sqft_min"] * df2["total_perc01"]

    # Bed quadrant credit (still out of 12; cols differ by round)
    bed_cols_r1 = [
        "r1_b1a", "r1_b1b", "r1_b1c", "r1_b1d",
        "r5_b2a", "r5_b2b", "r5_b2c", "r5_b2d",
        "r6_b3a", "r6_b3b", "r6_b3c", "r6_b3d"
    ]
    bed_cols_r2 = [
        "rd2_r1_b1a", "rd2_r1_b1b", "rd2_r1_b1c", "rd2_r1_b1d",
        "rd2_r6_b2a", "rd2_r6_b2b", "rd2_r6_b2c", "rd2_r6_b2d",
        "rd2_r7_b3a", "rd2_r7_b3b", "rd2_r7_b3c", "rd2_r7_b3d",
    ]
    bed_cols = bed_cols_r2 if is_r2 else bed_cols_r1

    for c in bed_cols:
        if c not in df2.columns:
            df2[c] = np.nan

    df2["bed_quadrants_searched"] = (
        df2[bed_cols]
        .apply(lambda s: _to_num(s), axis=0)
        .apply(lambda s: (s.fillna(0) != 0), axis=0)
        .sum(axis=1)
    )

    df2["percent_beds_searched"] = (df2["bed_quadrants_searched"] / 12) * 100
    return df2


group_order = ['Overall', 'Rescue Squads', 'Control Group','OIC', 'FF', 'Tech', 'RS-1', 'RS-2', 'RS-3', 'T-6', 'Other Assignment']
col1, col2 = st.columns(2)

with col1:
    member_id = st.selectbox("Select Member", sorted(df['id'].dropna().unique()), key="shared_member")

with col2:
    group_choice = st.selectbox("Compare To Group", group_order, key="shared_group")

st.markdown("---")

global_round = st.segmented_control(
    label="Round",
    options=["Round 1", "Round 2"],
    default="Round 1",
    key="global_round"
)
GLOBAL_IS_R2 = (global_round == "Round 2")

def participated_in_round(df, member_id, is_r2: bool) -> bool:
    """
    Returns True if participant has data for the selected round.
    """
    row = df[df["id"] == member_id]
    if row.empty:
        return False

    row = row.iloc[0]

    # Pick a reliable indicator column per round
    if is_r2:
        return pd.notna(row.get("rd2_total_ft_searched"))
    else:
        return pd.notna(row.get("total_ft_searched"))


def create_arrow_chart_spaced(member_est, compare_avg, actuals):
    trials = ["Trial 1", "Trial 2", "Trial 3"]
    group_order = ["ACTUAL", "YOU", "COMPARE"]
    gap = 5  # horizontal space between trials
    data = []
    trial_positions = []

    position = 0
    for i, trial in enumerate(trials):
        trial_positions.append((position, trial))
        for group, val, color in zip(["YOU", "COMPARE", "ACTUAL"],
                                     [member_est[i], compare_avg[i], actuals[i]],
                                     ["#F04923", "#00A86B", "#0067A5"]):
            data.append({
                "Group": group,
                "Start": position,
                "End": position + val,
                "LabelPos": position + val / 2,
                "Label": f"{int(round(val))} ft",
                "Color": color,
                "Trial": trial
            })

        position += max(member_est[i], compare_avg[i], actuals[i]) + gap
    
    trial_labels_df = pd.DataFrame([{
    "Trial": name,
    "XPos": pos + 6,
    } for pos, name in trial_positions])


    df = pd.DataFrame(data)

    base = alt.Chart(df)

    arrow_axis = alt.Axis(labels=False, ticks=False, title=None, domain=False, grid=True)

    arrows = base.mark_rule(tooltip=None).encode(
        x=alt.X('Start:Q', axis=arrow_axis),
        x2=alt.X2('End:Q'),
        y=alt.Y('Group:N', sort=group_order, title=None),
        color=alt.Color('Color:N', scale=None, legend=None),
        strokeWidth=alt.value(4)
    )

    heads = base.mark_point(shape='triangle-right', size=60, filled=True, tooltip=None).encode(
        x=alt.X('End:Q', axis=arrow_axis),
        y=alt.Y('Group:N', sort=group_order),
        color=alt.Color('Color:N', scale=None, legend=None),
    )

    labels = base.mark_text(
        align='center',
        dy=-10,
        fontSize=11,
        fontWeight='bold',
        tooltip=None
    ).encode(
        x=alt.X('LabelPos:Q', axis=arrow_axis),
        y=alt.Y('Group:N', sort=group_order),
        text='Label:N'
    )

    trial_labels = alt.Chart(trial_labels_df).mark_text(
    dy=-20,
    fontSize=13,
    fontWeight="bold"
    ).encode(
    x=alt.X("XPos:Q"),
    y=alt.value(0),
    text="Trial:N"
    )

    chart = (arrows + heads + labels + trial_labels).properties(
    width=500,
    height=300
    ).configure_title(anchor='start')

    return chart


def bordered_container(title, fig_or_chart, table_df):
    st.markdown(
        f"""
        <div style="
            background-color: #0067A5;
            color: white;
            border: 4px #0067A5;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        ">
        <h4 style="margin-top: 0; color: white;">{title}</h4>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        if isinstance(fig_or_chart, plt.Figure):
            st.pyplot(fig_or_chart, use_container_width=True)

        else:
            st.altair_chart(fig_or_chart, use_container_width=True)
        if table_df is not None:
            st.table(table_df.set_index("Metric"))
    st.markdown("</div>", unsafe_allow_html=True)


def show_task_metrics(df, member_id=None, group_choice=None, is_r2: bool = False):

    if member_id is None or group_choice is None:
        st.warning("Please select a member and comparison group.")
        return

    member_data = df[df['id'] == member_id]
    if member_data.empty:
        st.warning("No data available for this member.")
        return

    member_data = member_data.iloc[0]
    group_data_base = get_group_data(df, group_choice)
    if group_data_base.empty:
        st.warning("No comparison group found.")
        return

    def blue_task_header(title: str, subtitle: str | None = None):
        st.markdown(
            f"""
            <div style="
                background-color: #0067A5;
                color: white;
                border: 4px solid #0067A5;
                border-radius: 10px;
                padding: 10px;
                margin: 10px 0 10px 0;
            ">
                <h4 style="margin: 0; color: white;">{title}</h4>
                {f'<p style="margin:6px 0 0; font-size:14px; color:#f0f0f0; opacity:0.85; line-height:1.3;">{subtitle}</p>' if subtitle else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


    col1, col2 = st.columns(2)

    # --- Task 3: Random Distance ---
    with col1:
        blue_task_header("Random Travel Distance", "Participant asked to estimate travel distance")

        t3_cols = [pick_col("t3_s1_de", "rd2_t3_s1_de", is_r2),
                   pick_col("t3_s2_de", "rd2_t3_s2_de", is_r2),
                   pick_col("t3_s3_de", "rd2_t3_s3_de", is_r2)]
        member_est = [_to_num(member_data.get(c)) for c in t3_cols]
        compare_avg = [_to_num(group_data_base.get(c)).mean() for c in t3_cols]
        actuals = [10, 10, 15]

        if any(pd.isna(member_est)) or any(pd.isna(compare_avg)):
            st.warning("Missing data for Random Distance. Cannot render chart.")
        else:
            st.altair_chart(
                create_arrow_chart_spaced(member_est=member_est, compare_avg=compare_avg, actuals=actuals),
                use_container_width=True
            )

    # --- Task 4: Pre-Determined Distance ---
    with col2:
        blue_task_header("Pre-Determined Travel Distance", "Participant asked to travel 15 ft. forward")

        t4_cols = [pick_col("t4_s1_ad", "rd2_t4_s1_ad", is_r2),
                   pick_col("t4_s2_ad", "rd2_t4_s2_ad", is_r2),
                   pick_col("t4_s3_ad", "rd2_t4_s3_ad", is_r2)]
        member_est = [_to_num(member_data.get(c)) for c in t4_cols]
        compare_avg = [_to_num(group_data_base.get(c)).mean() for c in t4_cols]
        actuals = [15, 15, 15]

        if any(pd.isna(member_est)) or any(pd.isna(compare_avg)):
            st.warning("Missing data for Determined Distance. Cannot render chart.")
        else:
            st.altair_chart(
                create_arrow_chart_spaced(member_est=member_est, compare_avg=compare_avg, actuals=actuals),
                use_container_width=True
            )


    col3, col4 = st.columns(2)

    # --- Task 5: Triangle Completion ---
    with col3:
        blue_task_header("Triangle Completion Task","Participant asked to conduct a right-hand search, point to exit; then “beeline to exit”")

        t5_time_out = pick_col("t5_time_outbd", "rd2_t5_time_outbd", is_r2)
        t5_time_ret = pick_col("t5_time_rtrn", "rd2_t5_time_rtrn", is_r2)
        t5_bearing  = pick_col("t5_brng_angl", "rd2_t5_brng_angl", is_r2)
        t5_int_ang  = pick_col("t5_intsct_angl", "rd2_t5_intsct_angl", is_r2)

        actual_offset = _to_num(member_data.get(t5_bearing))
        intersect_angle = _to_num(member_data.get(t5_int_ang))

        ideal_relative = -25
        ideal_bearing = 270 - ideal_relative
        actual_bearing = 270 - actual_offset if pd.notna(actual_offset) else np.nan

        compare_offset = _to_num(group_data_base.get(t5_bearing)).mean()
        compare_bearing = 270 - compare_offset if pd.notna(compare_offset) else np.nan

        arrow_length = 3
        fig_triangle, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
        ax.set_aspect('equal')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.axis('off')
        ax.set_facecolor("#0067A5")

        ax.plot([3, 3], [0, 3], color='black', linewidth=2)
        ax.plot([0, 3], [3, 3], color='black', linewidth=2)

        for angle, color in zip([ideal_bearing, compare_bearing, actual_bearing], ['#0067A5', '#00A86B', '#F04923']):
            if pd.isna(angle):
                continue
            dx = arrow_length * np.cos(np.radians(angle))
            dy = arrow_length * np.sin(np.radians(angle))
            ax.arrow(0, 3, dx, dy, head_width=0.15, color=color, linewidth=2, length_includes_head=True)

        ax.plot([], [], color='#0067A5', label='Ideal')
        ax.plot([], [], color='#F04923', label='You')
        ax.plot([], [], color='#00A86B', label='Compare')

        legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
        for text_ in legend.get_texts():
            text_.set_color('black')

        st.pyplot(fig_triangle, use_container_width=True)

        def fmt(val, unit="", round_int=False):
            if pd.isna(val):
                return "—"
            return f"{int(round(val))}{unit}" if round_int else f"{val}{unit}"

        summary_df = pd.DataFrame({
            "Metric": ["Outbound Time", "Return Time", "Bearing Angle", "Intersect Angle"],
            "You": [
                fmt(_to_num(member_data.get(t5_time_out)), round_int=True),
                fmt(_to_num(member_data.get(t5_time_ret)), round_int=True),
                fmt(actual_offset, "°"),
                fmt(intersect_angle, "°"),
            ],
            "Compare": [
                fmt(_to_num(group_data_base.get(t5_time_out)).mean(), round_int=True),
                fmt(_to_num(group_data_base.get(t5_time_ret)).mean(), round_int=True),
                fmt(compare_offset, "°", round_int=True),
                fmt(_to_num(group_data_base.get(t5_int_ang)).mean(), "°", round_int=True),
            ]
        })
        st.table(summary_df.set_index("Metric"))

        # --- Task 6: Turn Direction and Veer ---
    with col4:
        blue_task_header(
    "Turn Direction and Veer Task",
    "Participant is asked to travel forward, stop, turn 90°, then continue forward."
)

        t6_angle = pick_col("t6_brng_angl", "rd2_t6_brng_angl", is_r2)
        t6_time  = pick_col("t6_tot", "rd2_t6_tot", is_r2)

        bearing_angle = _to_num(member_data.get(t6_angle))
        task_time = _to_num(member_data.get(t6_time))
        compare_angle = _to_num(group_data_base.get(t6_angle)).mean()

        fig_veer, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 5)
        ax.axis('off')
        ax.set_facecolor("#0067A5")

        arrow_length = 3
        origin_x, origin_y = -2, 0
        ax.plot([origin_x + 2, origin_x], [origin_y, origin_y], color='black', linewidth=2)

        for angle, color in zip([0, -compare_angle, -bearing_angle], ['#0067A5', '#00A86B', '#F04923']):
            if pd.isna(angle):
                continue
            dx = arrow_length * np.sin(np.radians(angle))
            dy = arrow_length * np.cos(np.radians(angle))
            ax.arrow(origin_x, origin_y, dx, dy, head_width=0.15, color=color, linewidth=2, length_includes_head=True)

        ax.plot([], [], color='#0067A5', label='Ideal')
        ax.plot([], [], color='#F04923', label='You')
        ax.plot([], [], color='#00A86B', label='Compare')

        legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, .5), frameon=False)
        for text_ in legend.get_texts():
            text_.set_color('black')

        st.pyplot(fig_veer, use_container_width=True)

        def fmt(val, unit="", round_int=False):
            if pd.isna(val):
                return "—"
            return f"{int(round(val))}{unit}" if round_int else f"{val}{unit}"

        metrics_df = pd.DataFrame({
            "Metric": ["Task Time", "Bearing Angle"],
            "You": [fmt(task_time), fmt(bearing_angle, "°")],
            "Compare": [fmt(_to_num(group_data_base.get(t6_time)).mean()), fmt(compare_angle, "°", round_int=True)]
        })
        st.table(metrics_df.set_index("Metric"))



def plot_score_distribution_kde(group_df, member_val, value_col, label):
    scores = pd.to_numeric(group_df[value_col], errors='coerce').dropna()
    if scores.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line(tooltip=None)
    # KDE setup
    kde = gaussian_kde(scores)
    x_vals = np.linspace(scores.min(), scores.max(), 200)
    y_vals = kde(x_vals)
    kde_df = pd.DataFrame({"Score": x_vals, "Density": y_vals})
    # Mean and labels
    group_mean = scores.mean()
    markers_df = pd.DataFrame({
        "Score": [member_val, group_mean],
        "Type": ["You", "Group Avg"],
        "Label": [f"Your score: {member_val:.1f}", f"Group average: {group_mean:.1f}"],
        "Color": ["#F04923", "#0067A5"]
    })

    # Shaded KDE area
    kde_area = alt.Chart(kde_df).mark_area(opacity=0.3, color="#0067A5", tooltip = None).encode(
        x="Score:Q",
        y=alt.Y("Density:Q", axis=None)
    )

    # KDE line
    kde_line = alt.Chart(kde_df).mark_line(color="#0067B9", tooltip = None).encode(
        x=alt.X("Score:Q", title=label),
        y=alt.Y("Density:Q", axis=None)
    )
    # Vertical lines with tooltips
    lines = alt.Chart(markers_df).mark_rule(strokeWidth=2).encode(
        x="Score:Q",
        color=alt.Color("Type:N", scale=alt.Scale(domain=["You", "Group Avg"], range=["#F04923", "#0067A5"]), legend = None)
    )
    # Inline labels (with different dy values for spacing)
    text_you = alt.Chart(markers_df[markers_df["Type"] == "You"]).mark_text(
        align="left", dx=3, dy=-10, fontSize=11, fontWeight="bold", color="#F04923", tooltip = None
    ).encode(
        x="Score:Q",
        y=alt.value(0),
        text="Label:N"
    )
    text_avg = alt.Chart(markers_df[markers_df["Type"] == "Group Avg"]).mark_text(
        align="left", dx=3, dy=-25, fontSize=11, fontWeight="bold", color="#0067A5", tooltip = None
    ).encode(
        x="Score:Q",
        y=alt.value(0),
        text="Label:N"
    )
    return (kde_area + kde_line + lines + text_you + text_avg).properties(width=300, height=250)

def calculate_metrics(df):
    df = df.copy()
    df['search_rate'] = (df['total_ft_searched'] / df['search_called']) * 60
    df['adjusted_score'] = df['search_rate'] * df['total_perc_searched']

    bed_cols = [
        "r1_b1a", "r1_b1b", "r1_b1c", "r1_b1d",
        "r5_b2a", "r5_b2b", "r5_b2c", "r5_b2d",
        "r6_b3a", "r6_b3b", "r6_b3c", "r6_b3d"
    ]
    df['bed_quadrants_searched'] = df[bed_cols].applymap(lambda x: x != 0 and pd.notna(x)).sum(axis=1)
    df['weighted_score'] = df['adjusted_score'] * (0.9 ** (12 - df['bed_quadrants_searched']))
    df['percent_beds_searched'] = df['bed_quadrants_searched'] / 12 * 100
    return df

def get_group_data(df, group_choice):
    group_choice = str(group_choice).strip()

    assignments = {"RS-1", "RS-2", "RS-3", "T-6", "Other Assignment"}
    rescue_squads = {"RS-1", "RS-2", "RS-3"}
    ranks = {"OIC", "FF", "Tech"}

    if group_choice == "Overall":
        return df
    elif group_choice == "Rescue Squads":
        return df[
            (df["assignment"].isin(rescue_squads)) |
            (df["id"] == 47)
        ]
    elif group_choice == "Control Group":
        return df[
            ~(df["assignment"].isin(rescue_squads)) &
            (df["id"] != 47)
        ]
    elif group_choice in assignments:
        if group_choice == "Other Assignment":
            return df[~df["assignment"].isin(assignments - {"Other Assignment"})]
        else:
            return df[df["assignment"] == group_choice]
    elif group_choice in ranks:
        return df[df["rank"] == group_choice]

    return pd.DataFrame()




def plot_interactive_distribution(group_df, member_val, value_col, label):
    df = group_df.copy()
    df = df[[value_col]].dropna().rename(columns={value_col: "Score"})
    df["Bin"] = pd.cut(df["Score"], bins=15)
    # Count frequencies per bin
    bin_counts = df.groupby("Bin").size().reset_index(name="Count")
    bin_counts["Bin Center"] = bin_counts["Bin"].apply(lambda x: x.mid)
    # Vertical lines for member and group average
    mean_val = group_df[value_col].mean()

    chart = alt.Chart(bin_counts).mark_bar(color="#E0E0E0", stroke="black").encode(
        x=alt.X("Bin Center:Q", title=label),
        y=alt.Y("Count:Q", title="Count"),
        tooltip=["Count", alt.Tooltip("Bin Center:Q", title="Score")]
    )

    member_line = alt.Chart(pd.DataFrame({"val": [member_val]})).mark_rule(color="#F04923", strokeWidth=2).encode(
        x="val:Q",
        tooltip=alt.Tooltip("val:Q", title="You")
    )

    mean_line = alt.Chart(pd.DataFrame({"val": [mean_val]})).mark_rule(color="#00A86B", strokeWidth=2).encode(
        x="val:Q",
        tooltip=alt.Tooltip("val:Q", title="Compare")
    )

    return (chart + member_line + mean_line).properties(height=300, width=300)



ZONE_COORDS_R2 = {
    # --- Room 1 ---
    "rd2_r1_b1a": (117, 256, 49, 71),
    "rd2_r1_b1b": (166, 255, 49, 72),
    "rd2_r1_b1c": (166, 328, 50, 71),
    "rd2_r1_b1d": (117, 327, 49, 72),
    "rd2_r1_c1":  (73, 38, 61, 119),
    "rd2_r1_c2":  [(141, 114), (163, 92), (190, 118), (167, 140)],
    "rd2_r1_c3":  [(170, 84), (198, 59), (223, 82), (194, 110)],
    "rd2_r1_c6":  (282, 357, 46, 43),
    "rd2_r1_f1":  [(135, 34), (203, 34), (159, 76), (181, 98), (176, 105), (161, 89), (135, 116)],
    "rd2_r1_f2":  [(49, 181), (51, 201), (196, 201), (196, 114), (182, 99), (176, 105),
                   (189, 117), (167, 141), (140, 114), (134, 117), (134, 181)],
    "rd2_r1_f3":  [(350, 126), (350, 175), (348, 127), (247, 127), (247, 80), (247, 35),
                   (205, 34), (248, 77), (204, 119), (197, 115), (197, 201), (303, 202),
                   (304, 194), (342, 195), (344, 203)],
    "rd2_r1_f4":  (197, 202, 107, 53),
    "rd2_r1_f5":  (216, 255, 88, 101),
    "rd2_r1_f6":  [(304, 255), (346, 255), (346, 295), (351, 295), (350, 344), (345, 344),
                   (346, 356), (304, 356)],
    "rd2_r1_f7":  (216, 355, 65, 47),
    "rd2_r1_f8":  [(49, 256), (49, 374), (103, 374), (104, 400), (116, 400), (116, 256)],
    "rd2_r1_f9":  (50, 202, 146, 53),

    # --- Room 2 ---
    "rd2_r2_c4": (380, 164, 130, 34),
    "rd2_r2_f1": (450, 35, 95, 83),
    "rd2_r2_f2": [(449, 117), (544, 118), (545, 130), (550, 127), (550, 178), (545, 179),
                  (545, 202), (511, 201), (510, 163), (449, 163)],
    "rd2_r2_f3": [(356, 119), (448, 118), (448, 163), (380, 163), (380, 201), (356, 201),
                  (355, 179), (351, 178), (350, 131), (355, 132)],

    # --- Room 3 ---
    "rd2_r3_t1": (427, 349, 112, 48),
    "rd2_r3_f1": [(471, 274), (433, 274), (432, 211), (407, 211), (407, 246), (368, 246),
                  (367, 211), (356, 211), (355, 292), (350, 293), (350, 342), (356, 342),
                  (426, 342), (471, 342)],
    "rd2_r3_f2": (471, 316, 74, 26),
    "rd2_r3_f3": (356, 343, 64, 59),

    # --- Room 4 ---
    "rd2_r4_s1": (555, 35, 78, 84),
    "rd2_r4_f1": [(653, 34), (638, 35), (636, 118), (644, 118), (645, 122), (695, 122),
                  (696, 118), (722, 118), (722, 34), (705, 33), (706, 72), (653, 72)],
    "rd2_r4_f2": (722, 35, 65, 25),
    "rd2_r4_f3": (722, 92, 66, 26),

    # --- Room 5 ---
    "rd2_r5_f1": [(551, 127), (645, 127), (645, 122), (696, 122), (696, 127), (696, 202),
                  (554, 202), (554, 179), (550, 178)],
    "rd2_r5_f2": [(699, 128), (844, 127), (844, 178), (839, 177), (840, 202), (766, 201),
                  (765, 205), (714, 205), (714, 201), (697, 201)],

    # --- Room 6 ---
    "rd2_r6_b2a": (687, 292, 74, 53),
    "rd2_r6_b2b": (762, 292, 73, 54),
    "rd2_r6_b2c": (762, 347, 73, 52),
    "rd2_r6_b2d": (687, 346, 74, 53),
    "rd2_r6_cl2": [(481, 211), (545, 211), (544, 237), (553, 235), (554, 285), (544, 284),
                   (545, 307), (481, 308)],
    "rd2_r6_f1": [(555, 212), (695, 211), (695, 292), (686, 291), (686, 308), (554, 307)],
    "rd2_r6_f2": [(696, 210), (712, 210), (714, 205), (765, 205), (764, 292), (696, 291)],
    "rd2_r6_f3": (765, 210, 73, 82),
    "rd2_r6_f4": [(555, 308), (686, 309), (686, 403), (595, 402), (595, 327), (554, 329)],

    # --- Room 7 ---
    "rd2_r7_b3a": (851, 256, 79, 52),
    "rd2_r7_b3b": (932, 256, 81, 53),
    "rd2_r7_b3c": (931, 309, 81, 54),
    "rd2_r7_b3d": (851, 309, 79, 54),
    "rd2_r7_c5":  (1022, 136, 36, 47),
    "rd2_r7_cl1": [(797, 35), (838, 36), (838, 61), (847, 63), (848, 113), (839, 114),
                   (838, 118), (797, 118)],
    "rd2_r7_f1":  [(848, 83), (969, 83), (970, 202), (848, 201), (847, 179), (845, 179),
                   (844, 127), (848, 127)],
    "rd2_r7_f2":  (848, 35, 59, 48),
    "rd2_r7_f3":  (1022, 36, 69, 81),
    "rd2_r7_f4":  (970, 84, 50, 118),
    "rd2_r7_f5":  (970, 203, 119, 51),
    "rd2_r7_f6":  (1013, 254, 35, 148),
    "rd2_r7_f7":  (1048, 339, 43, 64),
    "rd2_r7_f8":  (848, 364, 164, 39),
    "rd2_r7_f9":  [(848, 211), (847, 255), (970, 254), (969, 203), (898, 202), (898, 211)],
}


def draw_heatmaps_split(member_row, image_path="Floor Plan New.jpg"):
    try:
        base_img = Image.open(image_path).convert("RGBA")
    except (FileNotFoundError, UnidentifiedImageError):
        return None, None

    # two transparent overlays
    overlay_first = Image.new("RGBA", base_img.size, (255, 0, 0, 0))
    overlay_dup   = Image.new("RGBA", base_img.size, (255, 0, 0, 0))
    d1 = ImageDraw.Draw(overlay_first)
    d2 = ImageDraw.Draw(overlay_dup)
    zone_coords = {
        "r1_b1a": (132, 270, 52, 77),  # rectangle
        "r1_b1b": (185, 269, 51, 78),
        "r1_b1d": (133, 347, 51, 80),
        "r1_b1c": (184, 347, 52, 80),
        "r1_f1": [(81,116), (27,116), (27,27), (102,26), (103,98), (81,98)],  # polygon
        "r1_c1": (130, 48, 103, 50),
        "r1_cl1": [(341,428), (296,428), (296,375), (288,374), (288,272), (297,271), (297,218), (340,218)],
        "r1_f2": (83, 99, 100, 110),
        "r1_f3": [(183,99), (260,99), (261,91), (340,91), (341,128), (345,127), (345,183), (340,184), (340,209), (183,209)],
        "r1_f4": (183, 208, 104, 62),
        "r1_f5": (236, 270, 50, 157),
        "r1_f6": (27, 270, 105, 158),
        "r1_f7": [(29,234), (81,234), (83,209), (181,210), (182,269), (27,272)],
        "r2_f1": (451, 26, 99, 90),
        "r2_f2": [(550,183), (550,209), (475,212), (451,212),(451,116), (550,118), (554,128), (554,184)],
        "r2_f3": [(349,116), (449,116), (449,213), (422,213),(424,208), (351,208), (349,182), (344,180),(344,127), (349,127)],
        "r3_f1": [
        (351,219), (422,218), (422,213), (449,213), (449,322), (350,323),
        (351,314), (418,315), (418,277), (351,278)],
        "r3_f2": [
        (450,212), (479,212), (478,218), (553,219), (508,222),
        (508,265), (550,264), (550,323), (449,322)],
        "r3_f3": (450, 323, 100, 37),  # Rectangle
        "r3_f4": [
            (351,324), (449,323), (450,360), (417,360), (417,428), (350,428)
        ],
        "r3_t1": (425, 368, 117, 50),  # Rectangle

        # --- Room 4 ---
        "r4_c2": [(592,73), (626,53), (647,93), (613,110)],
        "r4_c3": [(795,52), (828,73), (806,110), (772,92)],
        "r4_f1": [
            (560,117), (709,117), (709,170), (693,169), (693,210),
            (668,209), (668,214), (612,214), (612,209), (561,210), (560,181)
        ],
        "r4_f2": [
            (709,118), (859,118), (859,128), (864,127), (864,183),
            (858,183), (858,208), (759,208), (760,170), (709,170)
        ],
        "r5_b2a": (698, 333, 79, 58),
        "r5_b2b": (698, 273, 79, 59),
        "r5_b2c": (777, 274, 77, 58),
        "r5_b2d": (777, 332, 77, 59),
        "r5_f1": (560, 218, 41, 74),
        "r5_f2": [
            (602,219), (612,219), (612,213), (668,213),
            (668,220), (698,219), (698,272), (698,323), (603,323)
        ],
        "r5_f3": (698, 220, 160, 53),
        "r5_f4": (698, 391, 161, 37),
        "r5_f5": (603, 325, 94, 103),
        "r5_f6": (560, 371, 43, 57),
        "r6_b3a": (964, 257, 58, 84),
        "r6_b3b": (1023, 257, 59, 84),
        "r6_b3c": (1023, 341, 59, 85),
        "r6_b3d": (964, 341, 59, 87),
        "r6_c4": (991, 51, 131, 65),
        "r6_c5": (1054, 137, 38, 51),
        "r6_cl2": [
            (870,220), (913,218), (913,273), (923,273),
            (923,376), (913,376), (913,428), (870,427)
        ],
        "r6_f1": [
            (864,127), (871,127), (870,122), (912,123),
            (913,117), (997,117), (997,209), (923,208),
            (870,208), (870,183), (864,183)
        ],
        "r6_f2": (913, 27, 53, 90),
        "r6_f3": (997, 117, 56, 92),
        "r6_f4": (997, 208, 129, 47),
        "r6_f5": (1082, 257, 44, 171),
        "r6_f6": (923, 257, 40, 171),
        "r6_f7": (923, 209, 72, 46)
        }
    

    for col_name, coords in zone_coords.items():
        base_val = member_row.get(col_name, 0)
        dup_val  = member_row.get(f"dup_{col_name}", 0)

        # First pass (light red)
        if pd.notna(base_val) and base_val != 0:
            if isinstance(coords, tuple):
                x, y, w, h = coords
                d1.rectangle([x, y, x + w, y + h], fill=(255, 0, 0, 100))
            else:
                d1.polygon(coords, fill=(255, 0, 0, 100))

        # Duplicates only (dark red)
        if pd.notna(dup_val) and dup_val != 0:
            if isinstance(coords, tuple):
                x, y, w, h = coords
                d2.rectangle([x, y, x + w, y + h], fill=(139, 0, 0, 160))
            else:
                d2.polygon(coords, fill=(139, 0, 0, 160))

    first_pass_img = Image.alpha_composite(base_img, overlay_first)
    duplicate_img  = Image.alpha_composite(base_img, overlay_dup)
    return first_pass_img, duplicate_img

def draw_heatmap_single(member_row, zone_coords, image_path, fill=(255, 0, 0, 100)):
    """
    Single-overlay heatmap (for Round 2).
    zone_coords values can be:
      - rect: (x, y, w, h)
      - poly: [(x1,y1), (x2,y2), ...]
    """
    try:
        base_img = Image.open(image_path).convert("RGBA")
    except (FileNotFoundError, UnidentifiedImageError):
        return None

    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    for col_name, coords in zone_coords.items():
        val = member_row.get(col_name, 0)
        if pd.notna(val) and val != 0:
            if isinstance(coords, tuple):
                x, y, w, h = coords
                d.rectangle([x, y, x + w, y + h], fill=fill)
            else:
                d.polygon(coords, fill=fill)

    return Image.alpha_composite(base_img, overlay)



def show_search_metrics(df, member_id=None, group_choice=None, is_r2: bool = False):

    if member_id is None or group_choice is None:
        st.warning("Please select both a member and a comparison group.")
        return



    st.markdown("---")


    df_round = add_search_derived(df, is_r2)
    member_row = df_round[df_round['id'] == member_id]
    if member_row.empty:
        st.warning("No data available for this member.")
        return
    member_row = member_row.iloc[0]

    group_data_base = get_group_data(df_round, group_choice)
    if group_data_base.empty:
        st.warning("No comparison group found.")
        return

    total_area = pick_total_area(is_r2)

    # ---- KPI cards ----
    searched_beds = int(_to_num(member_row.get("bed_quadrants_searched")))
    feet_val = _to_num(member_row.get("total_ft"))
    perc01 = _to_num(member_row.get("total_perc01"))

    feet_display = "—" if pd.isna(feet_val) else f"{round(feet_val):,} ft"
    percent_display = "—" if pd.isna(perc01) else f"{round(perc01 * 100)}%"

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">

    <div style="
        flex: 1;
        background-color: #0067A5;
        color: white;
        padding: 20px 10px;
        text-align: center;
        border-top-left-radius: 10px;
        border-bottom-left-radius: 10px;
        border: 4px solid #0067A5;
        border-right: none;
    ">
        <h2 style="margin:0; font-size:50px; color:white; ">{searched_beds}/12</h2>
        <p style="margin:0; font-size:16px;">Bed Quadrants Searched</p>
        <p style="
            margin:6px 12px 0;
            font-size:13px;
            line-height:1.2;
            color: #f0f0f0;
            opacity:0.8;
        ">
            Each bed was divided into 4 equal parts. You received credit for each quadrant that was adequately searched.
        </p>
    </div>

    <div style="
        flex: 1;
        background-color: #0067A5;
        color: white;
        padding: 20px 10px;
        text-align: center;
        border: 4px solid #0067A5;
        border-left: none;
        border-right: none;
    ">
        <h2 style="margin:0; font-size:50px; color:white; ">{feet_display}</h2>
        <p style="margin:0; font-size:16px;">Total Area Searched (Out of {total_area} ft²)</p>
    </div>

    <div style="
        flex: 1;
        background-color: #0067A5;
        color: white;
        padding: 20px 10px;
        text-align: center;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 10px;
        border: 4px solid #0067A5;
        border-left: none;
    ">
        <h2 style="margin:0; font-size:50px; color:white; ">{percent_display}</h2>
        <p style="margin:0; font-size:16px;">Total Percent Searched</p>
    </div>

    </div>
    """, unsafe_allow_html=True)

    # ---- Distributions ----
    search = _to_num(member_row.get("search_rate_sqft_min"))
    adjusted = _to_num(member_row.get("adjusted_score"))

    col1, col2 = st.columns(2)
    with col1:
        bordered_container(
            "Search Rate (sq ft/min)",
            plot_score_distribution_kde(group_data_base, search, "search_rate_sqft_min", "Search Rate (sq ft/min)"),
            None
        )

    with col2:
        st.markdown(
            """
            <div style="
                background-color: #0067A5;
                color: white;
                border: 4px #0067A5;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 20px;
            ">
                <h4 style="margin-top: 0; color: white;">Adjusted Score</h4>
                <p style="
                    font-size:16px;
                    color:#f0f0f0;
                    margin-top:-6px;
                    margin-bottom:10px;
                    opacity:0.85;
                ">
                    Search Rate × % of area covered on initial search
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.altair_chart(
            plot_score_distribution_kde(group_data_base, adjusted, "adjusted_score", "Adjusted Score"),
            use_container_width=True
        )

    # ---- Spatial Recall header + toggle in same row ----
    left, right = st.columns([5, 1.5], vertical_alignment="center")

    with left:
        st.markdown(
            """
            <div style="
                background-color: #0067A5;
                color: white;
                border: 4px solid #0067A5;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 10px;
            ">
                <h4 style="margin: 0; color: white;">Spatial Recall Task Heatmap</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

 
    # Heatmap-specific round toggle (independent)


    if not is_r2:
        first_img, dup_img = draw_heatmaps_split(member_row, image_path=ROUND1_IMG)
        if first_img is not None:
            st.image(first_img, caption="Round 1 — areas searched (light red)", use_container_width=True)
        else:
            st.warning("Round 1 floor plan image not found.")
        if dup_img is not None:
            st.image(dup_img, caption="Round 1 — duplicated areas (dark red)", use_container_width=True)
    else:
        r2_img = draw_heatmap_single(
            member_row,
            ZONE_COORDS_R2,
            image_path=ROUND2_IMG,
            fill=(255, 0, 0, 100)
        )
        if r2_img is not None:
            st.image(r2_img, caption="Round 2 — areas searched", use_container_width=True)
        else:
            st.warning("Round 2 floor plan image not found.")




    # ---- Notes (toggle matches section round) ----
    with st.expander("Notes", expanded=True):
        st.markdown("### Notes & Observations")

        def checkmark(val):
            if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
                return "—"
            s = str(val).strip().lower()
            if s in {"yes", "y", "true", "1"}:
                return "✅ Yes"
            if s in {"no", "n", "false", "0"}:
                return "❌ No"
            return str(val)  # fallback if it’s something like a comment


        missed_rooms_col = pick_col("missed_rooms", "rd2_missed_rooms", is_r2)
        disoriented_col  = pick_col("disoriented", "rd2_disoriented", is_r2)
        tool_col         = pick_col("tool", "rd2_tool", is_r2)
        duplicate_col    = pick_col("duplicate", "rd2_duplicate", is_r2)
        delayed_col      = pick_col("delayed_object", "rd2_delayed_object", is_r2)
        equip_col        = pick_col("equipment_issue", "rd2_equipment_issue", is_r2)
        furniture_col    = pick_col("furniture", "rd2_furniture", is_r2)
        notes_col        = pick_col("add_observations", "rd2_add_observations", is_r2)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Missed Rooms:**", checkmark(member_row.get(missed_rooms_col)))
            st.write("**Disoriented:**", checkmark(member_row.get(disoriented_col)))
            st.write("**Swept with Tool:**", checkmark(member_row.get(tool_col)))
        with col2:
            st.write("**Duplicate Rooms:**", checkmark(member_row.get(duplicate_col)))
            st.write("**Search Delayed by Object:**", checkmark(member_row.get(delayed_col)))
            st.write("**PPE/Equip Issues:**", checkmark(member_row.get(equip_col)))
        with col3:
            st.write("**Flipped/Moved Furniture:**", checkmark(member_row.get(furniture_col)))

        notes = member_row.get(notes_col, "")
        if notes and str(notes).strip():
            st.markdown("**Additional Observations:**")
            st.write(notes)


    # ---- Time to Bed (Round 1 only; Round 2 does not collect these fields) ----
    if not is_r2:
        bed_metrics = {
            "Time to Bed 1": "time_to_b1",
            "Time to Bed 2": "time_to_b2",
            "Time to Bed 3": "time_to_b3",
            "Time Called": "search_called"
        }

        chart_data = []
        for label, col in bed_metrics.items():
            chart_data.append({"Metric": label, "Value": _to_num(member_row.get(col)), "Type": "You"})
            chart_data.append({"Metric": label, "Value": round(_to_num(group_data_base.get(col)).mean()), "Type": "Compare"})

        df_chart = pd.DataFrame(chart_data)

        bar_chart = (
            alt.Chart(df_chart)
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=20),
                        sort=["Time to Bed 1", "Time to Bed 2", "Time to Bed 3", "Time Called"]),
                xOffset=alt.XOffset("Type:N"),
                y=alt.Y("Value:Q", title="Time (seconds)"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["You", "Compare"], range=["#F04923", "#0067A5"])),
                tooltip=["Type", "Value"]
            )
            .properties(width=500, height=300)
        )

        st.markdown(
            """
            <div style="
                background-color: #0067A5;
                color: white;
                border: 4px #0067A5;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 20px;
            ">
                <h4 style="margin-top: 0; color: white;">Time to Bed</h4>
                <p style="
                    font-size:14px;
                    color:#f0f0f0;
                    margin-top:-6px;
                    margin-bottom:10px;
                    line-height:1.3;
                    opacity:0.85;
                ">
                    Times are from start of search to 1st bed located, 2nd bed located, etc.
                    “Time Called” is from the start of evolution to the time the participant indicated search was complete.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.altair_chart(bar_chart, use_container_width=True)


    
def show_nasa_tlx(df, member_id=None, group_choice=None, is_r2: bool = False):
    if member_id is None or group_choice is None:
        st.warning("Please select a member and comparison group.")
        return

    member_data = df[df['id'] == member_id]
    if member_data.empty:
        st.warning("No data available for this member.")
        return
    member_row = member_data.iloc[0]

    group_data_base = get_group_data(df, group_choice)
    if group_data_base.empty:
        st.warning("No comparison group found.")
        return
    
    st.markdown("---")


    load_metrics = {
        "Mental": pick_col("t8_mental", "rd2_t8_mental", is_r2),
        "Physical": pick_col("t8_physical", "rd2_t8_physical", is_r2),
        "Temporal": pick_col("t8_temporal", "rd2_t8_temporal", is_r2),
        "Performance": pick_col("t8_performance", "rd2_t8_performance", is_r2),
        "Effort": pick_col("t8_effort", "rd2_t8_effort", is_r2),
        "Frustration": pick_col("t8_frustration", "rd2_t8_frustration", is_r2),
    }

    chart_data = []
    for label, col in load_metrics.items():
        chart_data.append({"Metric": label, "Value": _to_num(member_row.get(col)), "Type": "You"})
        chart_data.append({"Metric": label, "Value": _to_num(group_data_base.get(col)).mean(), "Type": "Compare"})

    ldi_df = pd.DataFrame(chart_data)

    bars = (
    alt.Chart(ldi_df)
    .mark_bar(tooltip=None)
    .encode(
        x=alt.X(
            "Metric:N",
            title=None,
            sort=list(load_metrics.keys()),
            axis=alt.Axis(labelAngle=20)
        ),
        xOffset=alt.XOffset("Type:N"),
        y=alt.Y("Value:Q", title="Rating (0–100)"),
        color=alt.Color(
            "Type:N",
            title="",
            legend=alt.Legend(
                orient="top",
                direction="horizontal",
                labelFontSize=12,
                symbolSize=120,
            ),
            scale=alt.Scale(
                domain=["You", "Compare"],
                range=["#F04923", "#0067A5"]
            )
        ),
        tooltip=alt.value(None),
    )
    .properties(width=500, height=300)
)

    labels = (
        alt.Chart(ldi_df)
        .mark_text(
            dy=-6,
            fontSize=11,
            fontWeight="bold",
            tooltip=None
        )
        .encode(
            x=alt.X("Metric:N", sort=list(load_metrics.keys())),
            xOffset=alt.XOffset("Type:N"),
            y=alt.Y("Value:Q"),
            text=alt.Text("Value:Q", format=".0f"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["You", "Compare"],
                    range=["#F04923", "#0067A5"]
                ),
                legend=None  # keep legend only on bars
            ),
            tooltip=alt.value(None),
        )
    )

    ldi_chart = bars + labels
    st.altair_chart(ldi_chart, use_container_width=True)


 
# --- Full Dashboard Page ---# --- Full Dashboard Page ---
def show_full_dashboard(df, member_id=None, group_choice=None, is_r2: bool = False):


    if member_id is None or group_choice is None:
        return

    # 🚨 Round participation guard
    if not participated_in_round(df, member_id, is_r2):
        st.markdown("##")
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 80px 20px;
                background-color: white;
            ">
                <h2 style="color:#333;">No data available</h2>
                <p style="font-size:18px; color:#666; max-width:600px; margin:auto;">
                    You did not participate in
                    <strong>Round 1</strong>.
                </p>
                <p style="font-size:15px; color:#888;">
                    Please switch to Round 2 to view your results.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return


    st.markdown("## Search Performance")
    show_search_metrics(df, member_id=member_id, group_choice=group_choice, is_r2=is_r2)

    st.markdown("---")

    st.markdown("## NASA Task Load Index")
    show_nasa_tlx(df, member_id=member_id, group_choice=group_choice, is_r2=is_r2)

    st.markdown("---")

    st.markdown(
        """
        <h2 style="margin-bottom:0;">Task Performance</h2>
        <p style="
            font-size:16px;
            color:#555;
            margin-top:4px;
            line-height:1.4;
            max-width:1200px;
        ">
            These tasks are typical of the kind of tests administered in Dr. Philbeck’s Movement Laboratory.
            We are interested in comparing how the firefighter population performs versus the general population.
        </p>
        """,
        unsafe_allow_html=True
    )



    st.markdown("---")

    show_task_metrics(
    df,
    member_id=member_id,
    group_choice=group_choice,
    is_r2=is_r2
)


show_full_dashboard(df, member_id=member_id, group_choice=group_choice, is_r2=GLOBAL_IS_R2)











