# Command-Line Execution
Using uv, all dependencies are resolved automatically without manual environment setup. To
render all eight clips into animations/:

```bash
uv run --with numpy --with scipy --with matplotlib --with imageio-ffmpeg pde_animations.py
```

To render a subset of clips by index (1–8), pass the desired IDs as command-line arguments
(for example, Clips 4, 5, and 7):

```bash
uv run --with numpy --with scipy --with matplotlib --with imageio-ffmpeg pde_animations.py 4 5 7
```

Alternatively, in a standard Python environment with numpy, scipy, matplotlib, and imageio-
ffmpeg (or system ffmpeg) installed, invoke python pde_animations.py [ids...].

# Common Customizations

• Changing Physical Parameters: Adjust c in wave_grid or k in heat_grid / boxcar_heat_grid
inside the relevant renderer or build_all. Note that changing c shifts the wavefront bound-
ary in Clip 4 to x= 1 + ct.

• Adding an Initial Velocity ψto Any Wave Clip: Pass both an exact antiderivative psi_antideriv
and a nonzero psi_scale to wave_grid (by default psi_scale=0.0, which disables the veloc-
ity term). If testing compactly supported propagation speed, choose a ψvanishing outside
[−1,1]—equivalently, an antiderivative that is constant outside [−1,1]—and raise z_max or
ylim to accommodate any overshoot above 1.

• Swapping Initial Profiles: Any vectorized function phi(x) compactly supported on [−1,1]
can be passed directly into wave_grid and heat_grid. For profiles supported on a wider
interval [−R,R], expand the direct-convolution integration bounds s = np.linspace(-R,
R, 400) inside heat_grid.
