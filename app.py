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
df = df_r1.merge(df_r2, on="id", how="left", suffixes=("", "_rd2dup"))

ROUND1_IMG = "Floor Plan New.jpg"
ROUND2_IMG = "Round 2 floor plan.png" 


# Page selector
st.set_page_config(layout="wide")

st.title("Wayfinding Performance Study Dashboard")
st.subheader("Initial Assessment")
st.markdown("Use the filters below to compare an individual member to a group average.")
st.markdown("---")

group_order = ['Overall', 'OIC', 'FF', 'Tech', 'RS-1', 'RS-2', 'RS-3', 'T-6', 'Other Assignment']
col1, col2 = st.columns(2)

with col1:
    member_id = st.selectbox("Select Member", sorted(df['id'].dropna().unique()), key="shared_member")

with col2:
    group_choice = st.selectbox("Compare To Group", group_order, key="shared_group")


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



def show_task_metrics(df, member_id=None, group_choice=None):

    if member_id is None or group_choice is None:
        st.warning("Please select a member and comparison group.")
        return

    member_data = df[df['id'] == member_id]
    if member_data.empty:
        st.warning("No data available for this member.")
        return

    group_data = get_group_data(df, group_choice)


    compare_avg = group_data[['t3_s1_de', 't3_s2_de', 't3_s3_de']].mean().tolist()


    #### Task 3: Random Distance ####
    member_est = member_data[['t3_s1_de', 't3_s2_de', 't3_s3_de']].values.flatten().tolist()
    compare_avg = group_data[['t3_s1_de', 't3_s2_de', 't3_s3_de']].mean().tolist()
    actuals = [10, 10, 15]

    if any(pd.isna(member_est)) or any(pd.isna(compare_avg)):
        st.warning("Missing data for Random Distance. Cannot render chart.")
        fig_random = None
    else:
        fig_random = create_arrow_chart_spaced(
            member_est=member_est,
            compare_avg=compare_avg,
            actuals=actuals
        )



    #### Task 4: Determined Distance ####
    member_est = member_data[['t4_s1_ad', 't4_s2_ad', 't4_s3_ad']].values.flatten().tolist()
    compare_avg = group_data[['t4_s1_ad', 't4_s2_ad', 't4_s3_ad']].mean().tolist()
    actuals = [15, 15, 15]

    if any(pd.isna(member_est)) or any(pd.isna(compare_avg)):
        st.warning("Missing data for Determined Distance. Cannot render chart.")
        fig_determined = None
    else:
        fig_determined = create_arrow_chart_spaced(
            member_est=member_est,
            compare_avg=compare_avg,
            actuals=actuals
        )


    #### Task 5: Triangle Completion ####
    actual_offset = member_data['t5_brng_angl'].values[0]
    intersect_angle = member_data['t5_intsct_angl'].values[0]
    ideal_relative = -25
    ideal_bearing = 270 - ideal_relative
    actual_bearing = 270 - actual_offset
    compare_offset = group_data['t5_brng_angl'].mean()
    compare_bearing = 270 - compare_offset

    arrow_length = 3
    fig_triangle, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.axis('off')

    ax.set_facecolor("#0067A5")
    ax.set_aspect('equal')
    ax.axis('off')
    ax.plot([3, 3], [0, 3], color='black', linewidth=2)
    ax.plot([0, 3], [3, 3], color='black', linewidth=2)

    for angle, color in zip([ideal_bearing, compare_bearing, actual_bearing], ['#0067A5', '#00A86B', '#F04923']):
        dx = arrow_length * np.cos(np.radians(angle))
        dy = arrow_length * np.sin(np.radians(angle))
        ax.arrow(0, 3, dx, dy, head_width=0.15, color=color, linewidth=2, length_includes_head=True)

    ax.plot([], [], color='#0067A5', label='Ideal')   
    ax.plot([], [], color='#F04923', label='You')
    ax.plot([], [], color='#00A86B', label='Compare')

    legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),frameon=False)
    for text in legend.get_texts():
        text.set_color('black')

    def fmt(val, unit="", round_int=False):
        if pd.isna(val):
            return "No time recorded"
        return f"{int(round(val))}{unit}" if round_int else f"{val}{unit}"

    group_data['t5_time_outbd'] = pd.to_numeric(group_data['t5_time_outbd'], errors='coerce')
    group_data['t5_time_rtrn'] = pd.to_numeric(group_data['t5_time_rtrn'], errors='coerce')
    group_data['t5_intsct_angl'] = pd.to_numeric(group_data['t5_intsct_angl'], errors='coerce')
    group_data['t5_brng_angl'] = pd.to_numeric(group_data['t5_brng_angl'], errors='coerce')

    summary_df = pd.DataFrame({
        "Metric": ["Outbound Time", "Return Time", "Bearing Angle", "Intersect Angle"],
        "You": [
            fmt(member_data['t5_time_outbd'].values[0]),
            fmt(member_data['t5_time_rtrn'].values[0]),
            fmt(actual_offset, "°"),
            fmt(intersect_angle, "°"),
        ],
        "Compare": [
        fmt(group_data['t5_time_outbd'].mean(), round_int= True),
        fmt(group_data['t5_time_rtrn'].mean(), round_int= True),
        fmt(compare_offset, "°", round_int= True),
        fmt(group_data['t5_intsct_angl'].mean(), "°", round_int= True),
    ]

    })

    #### Task 6: Turn and Veer ####
    bearing_angle = member_data['t6_brng_angl'].values[0]
    task_time = member_data['t6_tot'].values[0]
    compare_angle = group_data['t6_brng_angl'].mean()

    fig_veer, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 5)
    ax.axis('off')

    ax.set_facecolor("#0067A5")
    ax.set_aspect('equal')
    ax.axis('off')
    arrow_length = 3
    origin_x, origin_y = -2, 0
    ax.plot([origin_x + 2, origin_x], [origin_y, origin_y], color='black', linewidth=2)

    for angle, color in zip([0, -compare_angle, -bearing_angle], ['#0067A5', '#00A86B', '#F04923']):
        dx = arrow_length * np.sin(np.radians(angle))
        dy = arrow_length * np.cos(np.radians(angle))
        ax.arrow(origin_x, origin_y, dx, dy, head_width=0.15, color=color, linewidth=2, length_includes_head=True)

    ax.plot([], [], color='#0067A5', label='Ideal')
    ax.plot([], [], color='#F04923', label='You')
    ax.plot([], [], color='#00A86B', label='Compare')

    legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, .5),frameon=False)
    for text in legend.get_texts():
        text.set_color('black')


    metrics_df = pd.DataFrame({
        "Metric": ["Task Time", "Bearing Angle"],
        "You": [f"{task_time:.2f}", f"{bearing_angle:+.0f}°"],
        "Compare": [
        f"{group_data['t6_tot'].mean():.2f}",
        f"{compare_angle:+.0f}°"
    ]
    })

    #### Layout ####

    col1, col2 = st.columns(2)
    with col1:
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
                <h4 style="margin-top: 0; color: white;">Random Travel Distance</h4>
                <p style="
                    font-size:16px;
                    color:#f0f0f0;
                    margin-top:-6px;
                    margin-bottom:10px;
                    line-height:1.3;
                    opacity:0.85;
                ">
                    Participant asked to estimate travel distance
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if fig_random:
            st.altair_chart(fig_random, use_container_width=True)

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
                <h4 style="margin-top: 0; color: white;">Pre-Determined Travel Distance</h4>
                <p style="
                    font-size:16px;
                    color:#f0f0f0;
                    margin-top:-6px;
                    margin-bottom:10px;
                    line-height:1.3;
                    opacity:0.85;
                ">
                    Participant asked to travel 15 ft. forward
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if fig_determined:
            st.altair_chart(fig_determined, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
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
                <h4 style="margin-top: 0; color: white;">Triangle Completion Task</h4>
                <p style="
                    font-size:16px;
                    color:#f0f0f0;
                    margin-top:-6px;
                    margin-bottom:10px;
                    line-height:1.3;
                    opacity:0.85;
                ">
                    Participant asked to conduct a right-hand search, point to exit, then “beeline to exit.”
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if fig_triangle:
            st.pyplot(fig_triangle, use_container_width=True)
        if summary_df is not None:
            st.table(summary_df.set_index("Metric"))

    with col4:
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
                <h4 style="margin-top: 0; color: white;">Turn Direction and Veer Task</h4>
                <p style="
                    font-size:16px;
                    color:#f0f0f0;
                    margin-top:-6px;
                    margin-bottom:10px;
                    line-height:1.3;
                    opacity:0.85;
                ">
                    Participant asked to travel forward, stop, turn 90 degrees, then continue forward
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if fig_veer:
            st.pyplot(fig_veer, use_container_width=True)
        if metrics_df is not None:
            st.table(metrics_df.set_index("Metric"))



def plot_score_distribution_kde(group_df, member_val, value_col, label):
    scores = pd.to_numeric(group_df[value_col], errors='coerce').dropna()
    if scores.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()
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
    kde_area = alt.Chart(kde_df).mark_area(opacity=0.3, color="#0067A5").encode(
        x="Score:Q",
        y=alt.Y("Density:Q", axis=None)
    )

    # KDE line
    kde_line = alt.Chart(kde_df).mark_line(color="#0067B9").encode(
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
        align="left", dx=3, dy=-10, fontSize=11, fontWeight="bold", color="#F04923"
    ).encode(
        x="Score:Q",
        y=alt.value(0),
        text="Label:N"
    )
    text_avg = alt.Chart(markers_df[markers_df["Type"] == "Group Avg"]).mark_text(
        align="left", dx=3, dy=-25, fontSize=11, fontWeight="bold", color="#0067A5"
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
    ranks = {"OIC", "FF", "Tech"}

    if group_choice == "Overall":
        return df
    elif group_choice in assignments:
        if group_choice == "Other Assignment":
            return df[~df['assignment'].isin(assignments - {"Other Assignment"})]
        else:
            return df[df['assignment'] == group_choice]
    elif group_choice in ranks:
        return df[df['rank'] == group_choice]
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


def show_search_metrics(df, member_id=None, group_choice=None):

    if member_id is None or group_choice is None:
        st.warning("Please select both a member and a comparison group.")
        return

    df = calculate_metrics(df)
    member_row = df[df['id'] == member_id]
    if member_row.empty:
        st.warning("No data available for this member.")
        return

    group_data = calculate_metrics(get_group_data(df, group_choice))
    if group_data.empty:
        st.warning("No comparison group found.")
        return

    member_row = member_row.iloc[0]

    ## debugger####
    with st.expander("Debug Round 2", expanded=False):
        st.write("Round 2 cols present in df:", any(c.startswith("rd2_") for c in df.columns))
        st.write("Example rd2 columns:", [c for c in df.columns if c.startswith("rd2_")][:10])
        st.write(
            "Participant rd2 nonzero zones:",
            [k for k in ZONE_COORDS_R2.keys()
            if pd.to_numeric(member_row.get(k, 0), errors="coerce") not in [0, None, np.nan]][:20]
        )


    # Values
    searched = int(member_row['bed_quadrants_searched'])
    feet_display = f"{round(member_row['total_ft_searched']):,} ft"
    percent_display = f"{round(member_row['total_perc_searched'] * 100)}%"

    # Full connected row layout using flexbox
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">

    <!-- Card 1 -->
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
        <h2 style="margin:0; font-size:50px; color:white; ">{searched}/12</h2>
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

    <!-- Card 2 -->
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
        <p style="margin:0; font-size:16px;">Total Area Searched (Out of 515ft)</p>
    </div>

    <!-- Card 3 -->
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


    search = member_row['search_rate']
    adjusted = member_row['adjusted_score']
    # weighted = member_row['weighted_score']



    search_avg = group_data['search_rate'].mean()
    adjusted_avg = group_data['adjusted_score'].mean()
    # weighted_avg = group_data['weighted_score'].mean()

    
    # Get shared x-axis limits for consistent plot sizes
    xlims = {
        'search_rate': (group_data['search_rate'].min(), group_data['search_rate'].max()),
        'adjusted_score': (group_data['adjusted_score'].min(), group_data['adjusted_score'].max()),
        'weighted_score': (group_data['weighted_score'].min(), group_data['weighted_score'].max())
    }

    col1, col2 = st.columns(2)
    with col1:
        bordered_container(
            "Search Rate (sq ft/min)",
            plot_score_distribution_kde(group_data, search, "search_rate", "Search Rate (sq ft/min)"),
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
            plot_score_distribution_kde(group_data, adjusted, "adjusted_score", "Adjusted Score"),
            use_container_width=True
        )

        # Create a blank figure to pass into the bordered container
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.axis("off")  # Hide everything

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
    <h4 style="margin-top: 0; color: white;">Spatial Recall Task Heatmaps</h4>
    """,
    unsafe_allow_html=True
)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Round 1**")
        first_img, dup_img = draw_heatmaps_split(member_row, image_path=ROUND1_IMG)

        if first_img is not None:
            st.image(first_img, caption="Round 1 — areas searched (light red)", use_container_width=True)
        else:
            st.warning("Round 1 floor plan image not found.")

        if dup_img is not None:
            st.image(dup_img, caption="Round 1 — duplicated areas (dark red)", use_container_width=True)

    with colB:
        st.markdown("**Round 2**")
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

    st.markdown("</div>", unsafe_allow_html=True)




    # Interactive Bar Chart: Bed Times vs Group Avg
    bed_metrics = {
        "Time to Bed 1": "time_to_b1",
        "Time to Bed 2": "time_to_b2",
        "Time to Bed 3": "time_to_b3",
        "Time Called": "search_called"
    }

    chart_data = []
    for label, col in bed_metrics.items():
        chart_data.append({
            "Metric": label,
            "Value": member_row.get(col, np.nan),
            "Type": "You"
        })
        chart_data.append({
            "Metric": label,
            "Value": round(group_data[col].mean()),
            "Type": "Compare"
        })

    df_chart = pd.DataFrame(chart_data)

    bar_chart = (
    alt.Chart(df_chart)
    .mark_bar()
    .encode(
        x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=20),sort=["Time to Bed 1", "Time to Bed 2", "Time to Bed 3", "Time Called"],),
        xOffset=alt.XOffset("Type:N"),
        y=alt.Y("Value:Q", title="Time (seconds)"),
        color=alt.Color("Type:N", scale=alt.Scale(domain=["You", "Compare"],
                                          range=["#F04923", "#0067A5"])),

        tooltip=["Type", "Value"]
    )
    .properties(
        width=500,
        height=300
    )
)


    # Load Demand Index Comparison
    load_metrics = {
        "Mental": "t8_mental",
        "Physical": "t8_physical",
        "Temporal": "t8_temporal",
        "Performance": "t8_performance",
        "Effort": "t8_effort",
        "Frustration": "t8_frustration"
    }

    ldi_chart_data = []
    for label, col in load_metrics.items():
        ldi_chart_data.append({
            "Metric": label,
            "Value": member_row.get(col, np.nan),
            "Type": "You"
        })
        ldi_chart_data.append({
            "Metric": label,
            "Value": round(group_data[col].mean()),
            "Type": "Compare"
        })

    ldi_df = pd.DataFrame(ldi_chart_data)

    ldi_chart = (
        alt.Chart(ldi_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Metric:N",
                title=None,
                sort=["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"],  # Ordered categories
                axis=alt.Axis(labelAngle=20)
            ),
            xOffset=alt.XOffset("Type:N"),
            y=alt.Y("Value:Q", title="Rating (0–100)"),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["You", "Compare"],
                                          range=["#F04923", "#0067A5"])),

            tooltip=("Type", "Value")

        )
        .properties(
            width=500,
            height=300
        )
    )

    col1, col2 = st.columns(2)
    with col1:
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
                    Times are from start of search to 1st bed located, 2nd bed located, etc., not B1/B2/B3 from map above.
                    There will be a “time to bed” for each bed touched, even if it was not adequately searched.
                    “Time Called” is from the start of evolution to the time the participant indicated search was complete.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.altair_chart(bar_chart, use_container_width=True)

    with col2:
        bordered_container("NASA Task Load Index", ldi_chart, None)

    

    with st.expander("Notes", expanded=True):
        st.markdown("### Notes & Observations")

        def checkmark(val):
            return "✅ Yes" if str(val).strip().lower() == "yes" else "❌ No"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Missed Rooms:**", checkmark(member_row.get("missed_rooms")))
            st.write("**Disoriented:**", checkmark(member_row.get("disoriented")))
            st.write("**Swept with Tool:**", checkmark(member_row.get("tool")))
        with col2:
            st.write("**Duplicate Rooms:**", checkmark(member_row.get("duplicate")))
            st.write("**Search Delayed by Object:**", checkmark(member_row.get("delayed_object")))
            st.write("**PPE/Equip Issues:**", checkmark(member_row.get("equipment_issue")))
        with col3:
            st.write("**Flipped/Moved Furniture:**", checkmark(member_row.get("furniture")))

        notes = member_row.get('add_observations', "")
        if notes and str(notes).strip():
            st.markdown("**Additional Observations:**")
            st.write(notes)



# --- Full Dashboard Page ---
def show_full_dashboard(df, member_id=None, group_choice=None):
    st.markdown("## Search Performance")
    show_search_metrics(df, member_id=member_id, group_choice=group_choice)

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

    show_task_metrics(df, member_id=member_id, group_choice=group_choice)



show_full_dashboard(df, member_id=member_id, group_choice=group_choice)







