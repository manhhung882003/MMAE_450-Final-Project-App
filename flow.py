import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import joblib


# ============================================================
# Model architecture
# ============================================================
class FlowSurrogate(nn.Module):
    def __init__(self):
        super(FlowSurrogate, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# Load model and scalers once
# ============================================================
@st.cache_resource
def load_assets():
    model = FlowSurrogate()
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    return model, scaler_X, scaler_y


model, scaler_X, scaler_y = load_assets()


# ============================================================
# Prediction on a center-clustered masked Cartesian grid
# ============================================================
def predict_on_grid(Re_value, n_grid):
    # Square control volume: 50 m x 50 m centered at origin
    x_min, x_max = -25.0, 25.0
    y_min, y_max = -25.0, 25.0

    # Cylinder geometry
    R = 0.5  # radius = 0.5 m

    # Clustering parameter:
    # 1.0 -> uniform grid
    # >1.0 -> more points near x=0 and y=0
    cluster_strength = 3.0

    def clustered_coords(xmin, xmax, n, p):
        s = np.linspace(-1.0, 1.0, n)
        xi = np.sign(s) * np.abs(s)**p
        x = 0.5 * (xi + 1.0) * (xmax - xmin) + xmin
        return x

    x_vals = clustered_coords(x_min, x_max, n_grid, cluster_strength)
    y_vals = clustered_coords(y_min, y_max, n_grid, cluster_strength)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Mask out cylinder interior
    fluid_mask = (X**2 + Y**2) >= R**2

    # Evaluate only in fluid region
    x_fluid = X[fluid_mask]
    y_fluid = Y[fluid_mask]
    re_fluid = Re_value * np.ones_like(x_fluid)

    X_input = np.column_stack((x_fluid, y_fluid, re_fluid))
    X_scaled = scaler_X.transform(X_input)
    X_tensor = torch.FloatTensor(X_scaled)

    with torch.no_grad():
        y_scaled = model(X_tensor).numpy()

    y_pred = scaler_y.inverse_transform(y_scaled)

    # Fill full arrays with NaN inside cylinder
    U = np.full_like(X, np.nan, dtype=float)
    V = np.full_like(X, np.nan, dtype=float)
    P = np.full_like(X, np.nan, dtype=float)

    U[fluid_mask] = y_pred[:, 0]
    V[fluid_mask] = y_pred[:, 1]
    P[fluid_mask] = y_pred[:, 2]

    return X, Y, U, V, P, R


# ============================================================
# Streamlit page
# ============================================================
st.set_page_config(page_title="Cylinder Flow Surrogate", layout="wide")

st.title("Cylinder Flow Surrogate App")

st.write("Use the controls to choose the Reynolds number, field variable, and grid resolution.")
st.latex(r"\text{Inputs: } (x, y, Re)")
st.latex(r"\text{Outputs: } (u, v, p)")


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("Controls")

    field_name = st.selectbox(
        "Field to display",
        ["u", "v", "p", "velocity magnitude"]
    )

    Re_value = st.slider(
        "Reynolds number",
        min_value=10,
        max_value=100,
        value=50,
        step=1
    )

    n_grid = st.slider(
        "Grid resolution",
        min_value=101,
        max_value=2001,
        value=251,
        step=10
    )

    num_levels = st.slider(
        "Number of contour levels",
        min_value=10,
        max_value=60,
        value=30,
        step=5
    )


# ============================================================
# Run model
# ============================================================
X, Y, U, V, P, R = predict_on_grid(Re_value, n_grid)

velocity_mag = np.sqrt(U**2 + V**2)

if field_name == "u":
    Z = U
    title = "Predicted U-Velocity"
    cbar_label = "u"
elif field_name == "v":
    Z = V
    title = "Predicted V-Velocity"
    cbar_label = "v"
elif field_name == "p":
    Z = P
    title = "Predicted Pressure"
    cbar_label = "p"
else:
    Z = velocity_mag
    title = "Predicted Velocity Magnitude"
    cbar_label = r"$\sqrt{u^2 + v^2}$"


# ============================================================
# Metrics
# ============================================================
col1, col2, col3 = st.columns(3)
col1.metric("Re", f"{Re_value}")
col2.metric("Grid", f"{n_grid} x {n_grid}")
col3.metric("Field", field_name)


# ============================================================
# Plot contours
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

cf = ax.contourf(X, Y, Z, levels=num_levels, cmap="jet")
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(cbar_label)

# Draw cylinder boundary
theta = np.linspace(0.0, 2.0 * np.pi, 400)
x_cyl = R * np.cos(theta)
y_cyl = R * np.sin(theta)

ax.fill(x_cyl, y_cyl, color="white")
ax.plot(x_cyl, y_cyl, color="black", linewidth=1.5)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(title)
ax.set_aspect("equal")
ax.set_xlim([-12.5, 12.5])
ax.set_ylim([-12.5, 12.5])

st.pyplot(fig)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Grid Visualization", layout="wide")

st.title("Computational Grid Visualization")

with st.sidebar:
    st.header("Grid Controls")

    n_grid = st.slider("Grid resolution", 21, 301, 101, 20)

    x_min = st.number_input("x min", value=-25.0)
    x_max = st.number_input("x max", value=25.0)

    y_min = st.number_input("y min", value=-25.0)
    y_max = st.number_input("y max", value=25.0)

    R = st.number_input("Cylinder radius", value=0.5)

    show_points = st.checkbox("Show grid points", value=True)


# --------------------------------------------------
# Build clustered grid (same as NN evaluation)
# --------------------------------------------------
cluster_strength = st.slider(
    "Grid clustering near cylinder",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=0.5
)

def clustered_coords(xmin, xmax, n, p):
    s = np.linspace(-1.0, 1.0, n)
    xi = np.sign(s) * np.abs(s)**p
    x = 0.5 * (xi + 1.0) * (xmax - xmin) + xmin
    return x

x_vals = clustered_coords(x_min, x_max, n_grid, cluster_strength)
y_vals = clustered_coords(y_min, y_max, n_grid, cluster_strength)

X, Y = np.meshgrid(x_vals, y_vals)

# Mask cylinder interior
fluid_mask = (X**2 + Y**2) >= R**2


# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

# Optional: show full grid (before masking)
show_full = st.checkbox("Show full grid (including cylinder)", value=False)

if show_full:
    ax.scatter(
        X.ravel(),
        Y.ravel(),
        s=2,
        color="lightgray",
        alpha=0.4,
        label="Full grid"
    )

# Show fluid points only
if show_points:
    ax.scatter(
        X[fluid_mask],
        Y[fluid_mask],
        s=6,
        color="blue",
        label="Fluid points"
    )

# Draw cylinder
theta = np.linspace(0.0, 2*np.pi, 400)
x_cyl = R * np.cos(theta)
y_cyl = R * np.sin(theta)

ax.fill(x_cyl, y_cyl, color="white")
ax.plot(x_cyl, y_cyl, color="black", linewidth=2, label="Cylinder")

ax.set_aspect("equal")
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Clustered Grid with Cylinder Mask")

ax.legend()

st.pyplot(fig)
