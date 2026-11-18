# PyInstaller spec for the macOS Eyes.app -- run via packaging/build_mac_app.sh,
# not directly. PyInstaller resolves relative spec paths against the spec's
# own folder, so everything is anchored to the repo root explicitly.
import os
from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
datas = [(os.path.join(ROOT, "face_landmarker.task"), "."),
         (os.path.join(ROOT, "pose_landmarker.task"), ".")]
binaries = []
hiddenimports = ["glfw", "moderngl", "glcontext", "pythonosc"]

# mediapipe and moderngl ship native libs / data files PyInstaller's static
# analysis misses (mediapipe imports some modules lazily, moderngl loads its
# GL backend via glcontext at runtime).
for pkg in ("mediapipe", "moderngl", "glcontext", "glfw"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [os.path.join(ROOT, "eye_rig.py")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # jax/scipy/sounddevice come in via mediapipe submodules (genai, audio,
    # model_maker) this app never imports -- ~290MB of dead weight.
    # (matplotlib stays: mediapipe imports it at load time.)
    excludes=["tkinter", "IPython", "jupyter",
              "jax", "jaxlib", "scipy", "sounddevice", "_sounddevice_data"],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Eyes",
    console=False,
    target_arch="arm64",
)
coll = COLLECT(exe, a.binaries, a.datas, name="Eyes")

app = BUNDLE(
    coll,
    name="Eyes.app",
    bundle_identifier="com.jazergiles.eyes",
    info_plist={
        "CFBundleDisplayName": "Eyes",
        "CFBundleShortVersionString": "1.0.0",
        "NSHighResolutionCapable": True,
        # Required -- without it macOS kills the app (or hands it black
        # frames) the moment it opens the camera.
        "NSCameraUsageDescription":
            "Eyes uses the camera to track where you are so the eyes can follow you.",
    },
)
