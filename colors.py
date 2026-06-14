from matplotlib.colors import LinearSegmentedColormap

COLORS = {
    "STARBUCKS_GREEN": "#00704A",
    "COFFEE": "#4D2F1F",
    "MINT": "#D3F2E8",
    "O_ACCENT": "#FF8445",
    "Y_ACCENT": "#FCC756",
    "BEIGE": "#c9bd9f",
    "ROSE": "#f55b6f",
    "GREEN2": "#00D18B",
    "DARK_ROSE": "#f75267",
    "NEUTRAL_BG": "#f2f0df"
}

ACCENT_PALETTE = [
    COLORS["ROSE"],
    COLORS["STARBUCKS_GREEN"],
    COLORS["Y_ACCENT"],
    COLORS["O_ACCENT"]
]

STARBUCKS_COLORS = [
    COLORS["COFFEE"],
    COLORS["STARBUCKS_GREEN"],
    COLORS["GREEN2"],
    COLORS["Y_ACCENT"],
    COLORS["O_ACCENT"],
    COLORS["DARK_ROSE"],
    COLORS["MINT"],
    COLORS["BEIGE"]
]

SEGMENT_COLORS = {
    "Fast & Standard": COLORS["MINT"],
    "Patient & Standard": COLORS["BEIGE"],
    "Dissatisfied": COLORS["O_ACCENT"],
    "Bulk Buyers": COLORS["COFFEE"],
    "Customization Kings": COLORS["STARBUCKS_GREEN"]
}

STARBUCKS_CMAP = LinearSegmentedColormap.from_list(
    "Starbucks", 
    [COLORS["COFFEE"], COLORS["NEUTRAL_BG"], COLORS["STARBUCKS_GREEN"]]
)
