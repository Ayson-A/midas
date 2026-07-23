import streamlit as st
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from matplotlib import patheffects as pe

@st.cache_data
def load_data():
    generators = pd.read_csv("generators.csv")
    eventRisk = pd.read_csv("PJM_NRI_Counties.csv")
    fiber = pd.read_csv("Max_Fiber_County.csv")
    generationCapacity = pd.read_csv("Zone_Generation_Headroom.csv")
    generationCapacity["CleanEnergyRatio"] = generationCapacity["CleanEnergyRatio"] * 100
    LMP = pd.read_csv("LMP_stats.csv")
    waterStress = pd.read_csv("PJM_water_stress.csv")
    waterStress = waterStress.drop_duplicates(subset="County FIPS Code")
    developedLand = pd.read_csv("PJM_developed_land.csv")
    developedLand["Developed Land %"] = developedLand["Developed Land %"] * 100
    developedLand = developedLand.drop_duplicates(subset="County FIPS Code")

    zoneNameFixLMP = {"AECO": "AE", "PENELEC": "PENLC", "PPL": "PL", "PSEG": "PS"}
    LMP["pnode_name"] = LMP["pnode_name"].replace(zoneNameFixLMP)

    countyZoneMap = generators[["FIPS5", "Zone_21"]].drop_duplicates()
    countyZoneMap = countyZoneMap[~((countyZoneMap["FIPS5"] == 39053) & (countyZoneMap["Zone_21"] == "OVEC"))]  # 39053 = Gallia County FIPS
    countyZoneMap = countyZoneMap.drop_duplicates(subset="FIPS5")

    riskFiber = eventRisk.merge(fiber, left_on="STCOFIPS", right_on="FIPS5", how="inner")
    riskFiber = riskFiber.merge(countyZoneMap, left_on="STCOFIPS", right_on="FIPS5", how="inner")
    riskFiber = riskFiber.merge(generationCapacity, left_on="Zone_21", right_on=generationCapacity.columns[0], how="inner")
    riskFiber = riskFiber.merge(LMP, left_on="Zone_21", right_on="pnode_name", how="inner")
    riskFiber = riskFiber.merge(waterStress, left_on="STCOFIPS", right_on="County FIPS Code", how="left")
    riskFiber = riskFiber.merge(developedLand, left_on="STCOFIPS", right_on="County FIPS Code", how="left")

    countyTable = riskFiber[["COUNTY", "STATEABBRV", "Zone_21", "STCOFIPS", "RISK_SCORE",
                            "max_advertised_upload_speed", "mean", "std", "GenCapacity_ELCC", "Water Stress Level", "Developed Land %", "CleanEnergyRatio"]]
    
    countyTable = countyTable[countyTable["STCOFIPS"] != 37039]

    return countyTable

@st.cache_data
def load_geodata():
    # --- 1. LOAD GEOSPATIAL BASEMAP DATA ---
    print("Loading geospatial data...")
    us_states_path = Path("shapefiles", "cb_2018_us_state_500k.zip")
    us_states = gpd.read_file(us_states_path)
    states2drop_names = ["Alaska", "Hawaii", "Puerto Rico", "Commonwealth of the Northern Mariana Islands", "United States Virgin Islands", "American Samoa", "Guam"]
    us_states = us_states.loc[~us_states["NAME"].isin(states2drop_names)]
    us_counties_path = Path("shapefiles", "cb_2018_us_county_500k.zip")
    us_counties = gpd.read_file(us_counties_path).dropna()

    # --- Clip to PJM (state-boundary based) ---
    pjm_state_abbrevs = ['DE', 'IL', 'IN', 'KY', 'MD', 'MI', 'NJ', 'NC', 'OH', 'PA', 'VA', 'WV', 'DC']
    pjm_states = us_states[us_states['STUSPS'].isin(pjm_state_abbrevs)]
    inset_boundary = pjm_states.union_all()
    inset_states = pjm_states
    inset_counties = gpd.clip(us_counties, inset_boundary)

    # --- Fix: gpd.clip() can leave stray Point/MultiPoint geometries along the clip edge,
    # which render as dots instead of lines. Keep only real polygon geometries. ---
    print("Geometry types before filtering:")
    print("inset_states:", inset_states.geometry.type.value_counts().to_dict())
    print("inset_counties:", inset_counties.geometry.type.value_counts().to_dict())

    inset_states = inset_states[inset_states.geometry.type.isin(["Polygon", "MultiPolygon"])]
    inset_counties = inset_counties[inset_counties.geometry.type.isin(["Polygon", "MultiPolygon"])]

    # --- Load 9-zone shapefile ONLY to borrow its bounding box for a tight crop (not for shading) ---
    epa_ipm_shape_path = Path("shapefiles", "ipm_v6_regions.zip")
    epa = gpd.read_file(epa_ipm_shape_path, crs="EPSG:4326")
    epa = epa.to_crs(us_states.crs)
    inset_regions = epa[epa['IPM_Region'].str.startswith("PJM")].copy()

    bounds = inset_regions.total_bounds
    x_margin, y_margin = (bounds[2] - bounds[0]) * 0.1, (bounds[3] - bounds[1]) * 0.1

    return inset_states, inset_counties, bounds, x_margin, y_margin

def run_funnel(gen_pct, lmp_mean_pct, lmp_std_pct, risk_pct, fiber_pct, water_level, land_pct, clean_pct, countyTable, selected_states, grid_check, risk_check, fiber_check, water_check, land_check, clean_check):
    table = countyTable.copy()

    gencap_median = table["GenCapacity_ELCC"].quantile(gen_pct)
    lmp_mean_median = table["mean"].quantile(lmp_mean_pct)
    lmp_std_median = table["std"].quantile(lmp_std_pct)
    risk_median = table["RISK_SCORE"].quantile(risk_pct)
    fiber_median = table["max_advertised_upload_speed"].quantile(fiber_pct)
    land_median = table["Developed Land %"].quantile(land_pct)
    clean_median = table["CleanEnergyRatio"].quantile(clean_pct)

    if selected_states:
        table = table[table["STATEABBRV"].isin(selected_states)]

    if grid_check:
        table["passes_Grid"] = (
            (table["mean"] <= lmp_mean_median)
            & (table["std"] <= lmp_std_median)
            & (table["GenCapacity_ELCC"] >= gencap_median)
        )
    else:
        table["passes_Grid"] = True

    if risk_check:
        table["passes_Risk"] = table["RISK_SCORE"] <= risk_median
    else:
        table["passes_Risk"] = True

    if fiber_check:
        table["passes_Fiber"] = table["max_advertised_upload_speed"] >= fiber_median
    else:
        table["passes_Fiber"] = True

    if water_check:
        table["passes_Water"] = table["Water Stress Level"] <= water_level
    else:
        table["passes_Water"] = True

    if land_check:
        table["passes_Land"] = table["Developed Land %"] >= land_median
    else:
        table["passes_Land"] = True

    if clean_check:
        table["passes_Clean"] = table["CleanEnergyRatio"] >= clean_median
    else:
        table["passes_Clean"] = True

    funnel_counts = [("Start", len(table))]
    survivors = table.copy()

    if grid_check:
        survivors = survivors[survivors["passes_Grid"]]
        funnel_counts.append(("Grid", len(survivors)))

    if risk_check:
        survivors = survivors[survivors["passes_Risk"]]
        funnel_counts.append(("Risk", len(survivors)))

    if fiber_check:
        survivors = survivors[survivors["passes_Fiber"]]
        funnel_counts.append(("Fiber", len(survivors)))

    if water_check:
        survivors = survivors[survivors["passes_Water"]]
        funnel_counts.append(("Water", len(survivors)))

    if land_check:
        survivors = survivors[survivors["passes_Land"]]
        funnel_counts.append(("Land", len(survivors)))

    if clean_check:
        survivors = survivors[survivors["passes_Clean"]]
        funnel_counts.append(("Clean", len(survivors)))

    return funnel_counts, survivors, table

def build_remaining_map(table, inset_counties, inset_states, bounds, x_margin, y_margin):
    fig, ax = plt.subplots(figsize=(14, 10))
    inset_counties.boundary.plot(ax=ax, edgecolor="grey", linewidth=0.2, zorder=2)

    table = table.copy()
    table["STCOFIPS"] = table["STCOFIPS"].astype(str).str.zfill(5)

    in_pool_fips = set(table["STCOFIPS"])
    survivor_fips = set(table[table["passes_Grid"] & table["passes_Risk"] & table["passes_Fiber"] & table["passes_Water"] & table["passes_Land"] & table["passes_Clean"]]["STCOFIPS"])

    in_pool_mask = inset_counties["GEOID"].isin(in_pool_fips)
    alive_mask = inset_counties["GEOID"].isin(survivor_fips)

    dead = inset_counties[in_pool_mask & ~alive_mask]
    alive = inset_counties[alive_mask]

    if len(dead) > 0:
        dead.plot(ax=ax, color="lightgrey", zorder=1, alpha=0.7)
    if len(alive) > 0:
        alive.plot(ax=ax, color="green", zorder=1, alpha=0.7)

    inset_states.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, zorder=3)

    for idx, row in inset_states.iterrows():
        point = row.geometry.representative_point()
        if (
            'STUSPS' in inset_states.columns
            and row.geometry.area > 0
            and bounds[0] - x_margin <= point.x <= bounds[2] + x_margin
            and bounds[1] - y_margin <= point.y <= bounds[3] + y_margin
        ):
            ax.text(
                point.x, point.y, row['STUSPS'],
                fontsize=10, fontweight='bold', ha='center', va='center', color='black',
                path_effects=[pe.withStroke(linewidth=3, foreground='white', alpha=0.7)]
            )

    ax.set_xlim(bounds[0] - x_margin, bounds[2] + x_margin)
    ax.set_ylim(bounds[1] - y_margin, bounds[3] + y_margin)
    ax.set_axis_off()

    return fig

def build_line_chart(funnel_counts):
    stages, counts = zip(*funnel_counts)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(stages, counts, marker="o")
    ax.set_ylabel("Number of Counties Remaining")
    ax.set_xlabel("Filter Stage")
    
    return fig

@st.cache_data
def build_distribution_maps(table, _inset_counties, _inset_states, bounds, x_margin, y_margin):
    table = table.copy()
    table["STCOFIPS"] = table["STCOFIPS"].astype(str).str.zfill(5)
    bubble_counties = _inset_counties.merge(
        table[["STCOFIPS", "GenCapacity_ELCC", "RISK_SCORE", "max_advertised_upload_speed", "Water Stress Level", "Developed Land %", "CleanEnergyRatio"]],
        left_on="GEOID", right_on="STCOFIPS", how="left"
    )
    bubble_counties["centroid"] = bubble_counties.geometry.centroid

    # Criteria config: (Column, Label, Cap Outliers, Color Palette, Custom Bins)
    criteria = [
        ("GenCapacity_ELCC", "Generation Capacity (ELCC-weighted MW)", False,
         ['#93c5fd', '#3b82f6', '#1d4ed8', '#1e3a8a'], None),
        ("RISK_SCORE", "FEMA NRI Risk Score", False,
         ['#fde68a', '#f59e0b', '#b45309', '#78350f'], None),
        ("max_advertised_upload_speed", "Max Fiber Upload Speed (Mbps)", False,
         ['#a7f3d0', '#34d399', '#059669', '#064e3b'], None),
        ("Water Stress Level", "Water Stress Level", False, 
         ['#f3e8ff', '#c084fc', '#9333ea', '#6b21a8', '#3b0764'], 
         [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),  # Bin edges centered around integers 1 to 5
        ("Developed Land %", "Developed Land %", False,
         ['#fecaca', '#f87171', '#dc2626', '#7f1d1d'], None),
        ("CleanEnergyRatio", "Clean Energy Ratio", False, 
         ['#a5f3fc', '#22d3ee', '#0891b2', '#164e63'], None)
    ]

    figures = []

    for col, label, cap_outliers, bin_colors, custom_bins in criteria:
        fig_b, ax = plt.subplots(figsize=(14, 10))

        bubble_counties.boundary.plot(ax=ax, edgecolor="grey", linewidth=0.2, zorder=2)
        _inset_states.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=3)

        valid = bubble_counties.dropna(subset=[col]).copy()

        if cap_outliers:
            cap_value = valid[col].quantile(0.95)
            valid["plot_value"] = valid[col].clip(upper=cap_value)
        else:
            valid["plot_value"] = valid[col]

        # Discrete handling vs Continuous quantile handling
        if custom_bins is not None:
            bin_edges = custom_bins
            # Maps bin [0.5, 1.5] -> Level 1, [1.5, 2.5] -> Level 2, etc.
            bin_labels = [f"Level {i+1}" for i in range(len(bin_edges) - 1)]
            bin_sizes = [40, 100, 180, 280, 400]
        else:
            raw_quantiles = valid["plot_value"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
            bin_edges = np.unique(raw_quantiles)
            bin_labels = []
            for i in range(len(bin_edges) - 1):
                lower = int(bin_edges[i]) if i == 0 else int(bin_edges[i]) + 1
                upper = int(bin_edges[i + 1])
                bin_labels.append(f"{lower} - {upper}")
            bin_sizes = [40, 120, 250, 450]

        valid["category"] = pd.cut(
            valid["plot_value"], 
            bins=bin_edges, 
            labels=bin_labels, 
            include_lowest=True
        )

        # Plot centroids grouped by bin
        for i, cat_label in enumerate(bin_labels):
            subset = valid[valid["category"] == cat_label]
            if not subset.empty:
                ax.scatter(
                    subset["centroid"].x, subset["centroid"].y,
                    s=bin_sizes[i % len(bin_sizes)], 
                    color=bin_colors[i % len(bin_colors)],
                    edgecolor="white", linewidth=0.5, alpha=0.85, zorder=4
                )

        # Overlay state labels
        for idx, row in _inset_states.iterrows():
            point = row.geometry.representative_point()
            if (
                'STUSPS' in _inset_states.columns
                and row.geometry.area > 0
                and bounds[0] - x_margin <= point.x <= bounds[2] + x_margin
                and bounds[1] - y_margin <= point.y <= bounds[3] + y_margin
            ):
                ax.text(
                    point.x, point.y, row['STUSPS'],
                    fontsize=11, fontweight='bold', ha='center', va='center', color='black',
                    path_effects=[pe.withStroke(linewidth=3, foreground='white', alpha=0.7)]
                )

        # Build legend
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label=bin_labels[i],
                markerfacecolor=bin_colors[i % len(bin_colors)], markeredgecolor='k',
                markersize=16 + i * 4)
            for i in range(len(bin_labels))
        ]
        legend = ax.legend(handles=legend_handles, loc='lower left', title=label, fontsize=18, labelspacing=1.2)
        legend.get_title().set_fontsize(20)

        ax.set_title(label, fontsize=16)
        ax.set_xlim(bounds[0] - x_margin, bounds[2] + x_margin)
        ax.set_ylim(bounds[1] - y_margin, bounds[3] + y_margin)
        ax.set_axis_off()

        plt.tight_layout()
        figures.append(fig_b)

    return figures

st.set_page_config(
    page_title="PJM Data Center Siting Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("MIDAS: Data Center Siting Within PJM")
st.caption("**MIDAS** (Multi-criteria Interactive Data center Alignment and Siting)")
st.sidebar.title("Criteria")

countyTable = load_data()
total_count = len(countyTable)
inset_states, inset_counties, bounds, x_margin, y_margin = load_geodata()

available_states = sorted(countyTable["STATEABBRV"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Filter by State(s)", 
    options=available_states, 
    default=available_states # Defaults to showing all states so the map isn't blank
)

grid_check = st.sidebar.checkbox(
    "Grid",
    help="Generation Capacity: local electricity supply within a zone (Source: ICARUS-PJM). LMP: price of electricity at a location, higher values signal grid congestion (Source: PJM Data Miner)."
)
gen_capacity_slider = st.sidebar.slider("Generation Capacity", 0.0, 1.0, 0.5)
raw_gen = countyTable["GenCapacity_ELCC"].quantile(gen_capacity_slider)
st.sidebar.caption(f"**Minimum Generation Capacity:** {raw_gen:.0f} MW")

lmp_mean_slider = st.sidebar.slider("Locational Marginal Pricing Mean", 0.0, 1.0, 0.5)
raw_lmp_mean = countyTable["mean"].quantile(lmp_mean_slider)
st.sidebar.caption(f"**Maximum Average Price:** ${raw_lmp_mean:.2f} / MWh")

lmp_std_slider = st.sidebar.slider("Locational Marginal Pricing STD", 0.0, 1.0, 0.5)
raw_lmp_std = countyTable["std"].quantile(lmp_std_slider)
st.sidebar.caption(f"**Maximum Volatility:** ${raw_lmp_std:.2f} / MWh")

st.sidebar.divider()

risk_check = st.sidebar.checkbox(
    "Risk",
    help="FEMA National Risk Index represents the potential for negative impacts resulting from natural hazards. Lower scores mean lower risk. Source: FEMA NRI."
)
risk_slider = st.sidebar.slider("National Risk Index", 0.0, 1.0, 0.5)
raw_risk = countyTable["RISK_SCORE"].quantile(risk_slider)
st.sidebar.caption(f"**Maximum Risk Score:** {raw_risk:.2f}")

st.sidebar.divider()

fiber_check = st.sidebar.checkbox(
    "Fiber",
    help="Maximum advertised fiber upload speed available in the county for businesses. Source: FCC National Broadband Map."
)
fiber_slider = st.sidebar.slider("Max Fiber Speed", 0.0, 1.0, 0.5)
raw_fiber = countyTable["max_advertised_upload_speed"].quantile(fiber_slider)
st.sidebar.caption(f"**Minimum Upload:** {raw_fiber:.0f} Mbps")

with st.sidebar.expander("Additional Stages (Experimental)"):
    water_check = st.checkbox(
        "Water Stress",
        help="Baseline water stress: ratio of total water demand (domestic, industrial, irrigation, livestock) to available renewable surface and groundwater supplies. Higher values indicate more competition among users. Source: WRI Aqueduct 4.0."
    )
    water_slider = st.slider(
        "Max Allowed Water Stress Level",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )

    st.divider()

    land_check = st.checkbox(
        "Land",
        help="Percentage of land already developed in the county, indicating available space for new construction. Source: MRLC National Land Cover Database (NLCD)."
    )
    land_slider = st.slider("Developed Land %", 0.0, 1.0, 0.5)
    raw_land = countyTable["Developed Land %"].quantile(land_slider)
    st.caption(f"**Minimum Developed Land %:** {raw_land:.2f}")

    st.divider()

    clean_check = st.checkbox(
        "Clean Energy",
        help="Share of a zone's ELCC-weighted generation capacity from clean sources (renewables and nuclear) relative to total capacity. Source: ICARUS-PJM."
    )
    clean_slider = st.slider("Clean Energy Ratio", 0.0, 1.0, 0.5)
    raw_clean = countyTable["CleanEnergyRatio"].quantile(clean_slider)
    st.caption(f"**Minimum Clean Energy Ratio %:** {raw_clean:.2f}")

generate_map_button = st.sidebar.button("Generate Map")

tab1, tab2, tab3, tab4 = st.tabs(["Remaining Counties", "Distribution Maps", "Sources", "About"])
with tab1:
    if generate_map_button:
        # Run your funnel logic with the selected states and criteria
        funnel_counts, survivors, table = run_funnel(
            gen_capacity_slider, lmp_mean_slider, lmp_std_slider, 
            risk_slider, fiber_slider, water_slider, land_slider, clean_slider,
            countyTable, selected_states, 
            grid_check, risk_check, fiber_check, water_check, land_check, clean_check
        )
        
        with st.container(border=True):
            st.subheader("Remaining Counties Map")
            st.caption(f"**{len(survivors)}** Counties Remaining")
            remaining_map_fig = build_remaining_map(table, inset_counties, inset_states, bounds, x_margin, y_margin)
            st.pyplot(remaining_map_fig)

        with st.container(border=True):
            st.subheader("Remaining Counties Progression")
            remaining_line_fig = build_line_chart(funnel_counts)
            st.pyplot(remaining_line_fig)

        with st.container(border=True):
            st.subheader("Surviving County Table")
            st.dataframe(survivors, use_container_width=True)
            
            st.download_button(
                label="Download Filtered CSV", 
                data=survivors.to_csv(index=False).encode("utf-8"), 
                file_name="pjm_siting_survivors.csv",
                mime="text/csv"
            )

with tab2:
    distribution_figs = build_distribution_maps(countyTable, inset_counties, inset_states, bounds, x_margin, y_margin)
    for fig in distribution_figs:
        st.pyplot(fig)
        st.divider()

with tab3:
    st.subheader("Data Sources")
    st.markdown("""
    **Aqueduct Water Risk Atlas.** (n.d.). Retrieved July 9, 2026, from [wri.org/applications/aqueduct/water-risk-atlas](https://www.wri.org/applications/aqueduct/water-risk-atlas/)

    **Data Miner 2.** (n.d.). Retrieved July 9, 2026, from [dataminer2.pjm.com/feed/da_hrl_lmps/definition](https://dataminer2.pjm.com/feed/da_hrl_lmps/definition)

    **FCC National Broadband Map.** (n.d.). Retrieved July 9, 2026, from [broadbandmap.fcc.gov](https://broadbandmap.fcc.gov)

    **HopkinsICARUS.** (2026). *HopkinsICARUS/ICARUS-PJM-Dataset* [Python]. [github.com/HopkinsICARUS/ICARUS-PJM-Dataset](https://github.com/HopkinsICARUS/ICARUS-PJM-Dataset) (Original work published 2025). Developed by Saroj Khanal.

    **MRLC Viewer.** (n.d.). Retrieved July 9, 2026, from [mrlc.gov/viewer](https://www.mrlc.gov/viewer/)

    **National Risk Index for Natural Hazards.** FEMA.gov. (2026, January 7). [fema.gov/flood-maps/products-tools/national-risk-index](https://www.fema.gov/flood-maps/products-tools/national-risk-index)
    """)

with tab4:
    st.subheader("About MIDAS")
    st.markdown("""
    MIDAS (Multi-criteria Interactive Data center Alignment and Siting) was developed as part of the 
    ROSETAS REU Summer 2026 program at Johns Hopkins University's Ralph O'Connor Sustainable 
    Energy Institute.

    This tool brings together the data collected and analyzed over the course of the project into a 
    single interactive web application. It allows a user to filter and explore PJM's 312 counties 
    across a range of siting criteria, including grid characteristics, risk to natural hazards, fiber 
    infrastructure, and additional experimental stages like water stress, developed land, and clean energy.

    A prospective data center developer can use MIDAS to explore which counties meet their specific 
    priorities, since it is up to the user to decide which criteria matter most for their own siting 
    decision. The tool also includes distribution maps showing how each individual criterion varies 
    across PJM, and a downloadable CSV of the current filtered county list for further use.
    """)

    st.divider()

    st.subheader("Acknowledgements")
    st.markdown("""
    I would like to thank and acknowledge my mentors, Saroj Khanal and Zhiyi Zhou, for their guidance, 
    feedback, and support throughout this project. I am also grateful to my PIs, Professors Dennice 
    Gayme and Yury Dvorkin, for their sponsorship and direction of my project.

    This work was supported by the National Science Foundation (EEC-2349378), as part of the REU 
    program at Johns Hopkins University's Ralph O'Connor Sustainable Energy Institute.

    For more information, contact aayson1@jh.edu.
    """)
