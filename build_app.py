"""
Build "Filter Portal.app"
=========================

Makes a normal Mac app, so you can start Filter Portal from the Dock,
Spotlight or Launchpad instead of the terminal:

    .venv/bin/python build_app.py

It creates ~/Applications/Filter Portal.app and adds it to the Dock (skip
that with --no-dock). The app runs filter_portal.py from THIS folder, so if
you move or rename the folder, run this again from its new place.

How the app works:
- A small compiled launcher (Swift) first asks macOS for camera access the
  official way and waits for your answer. If access is off, it offers a
  button that opens the right page in System Settings.
- It then runs Resources/run.sh, which sets up Python if .venv is missing
  (needs internet, takes a few minutes the first time) and starts
  filter_portal.py.
- If something goes wrong you get a message box; details are logged to
  ~/Library/Logs/Filter Portal.log
"""

import argparse
import os
import pathlib
import platform
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

APP_NAME = "Filter Portal"
BUNDLE_ID = "local.filterportal"
EXECUTABLE = "FilterPortal"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_APP_PATH = os.path.expanduser(f"~/Applications/{APP_NAME}.app")
LSREGISTER = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/"
              "LaunchServices.framework/Support/lsregister")
CAMERA_USAGE = ("Filter Portal uses the camera to track your hands and show "
                "the live video with filters.")

# --------------------------------------------------------------------------
# The launcher program (Swift). It's the app's main program.
# --------------------------------------------------------------------------

LAUNCHER_SWIFT = r"""
// Filter Portal launcher, compiled by build_app.py.
//  1. Makes sure macOS lets this app use the camera: asks once
//     ("allow camera access?") and waits for your answer.
//  2. Runs Resources/run.sh, which starts filter_portal.py.
import AppKit
import AVFoundation

func statusName(_ status: AVAuthorizationStatus) -> String {
    switch status {
    case .authorized: return "authorized"
    case .notDetermined: return "not decided yet"
    case .denied: return "denied"
    case .restricted: return "restricted"
    @unknown default: return "unknown"
    }
}

let runScript = Bundle.main.path(forResource: "run", ofType: "sh")

// Self-check used by build_app.py: never asks anything, never uses the camera.
if CommandLine.arguments.contains("--check") {
    print("run.sh: \(runScript ?? "MISSING")")
    print("camera permission: "
          + statusName(AVCaptureDevice.authorizationStatus(for: .video)))
    exit(runScript == nil ? 1 : 0)
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)

@discardableResult
func showAlert(_ title: String, _ message: String,
               _ buttons: [String] = ["OK"]) -> NSApplication.ModalResponse {
    let alert = NSAlert()
    alert.messageText = title
    alert.informativeText = message
    for button in buttons { alert.addButton(withTitle: button) }
    if #available(macOS 14, *) { app.activate() }
    else { app.activate(ignoringOtherApps: true) }
    return alert.runModal()
}

final class Answer: @unchecked Sendable { var granted = false }

func cameraAllowed() -> Bool {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
        return true
    case .notDetermined:
        // Shows macOS's "allow camera access?" question and waits for it.
        let answer = Answer()
        let done = DispatchSemaphore(value: 0)
        AVCaptureDevice.requestAccess(for: .video) { granted in
            answer.granted = granted
            done.signal()
        }
        done.wait()
        return answer.granted
    default:
        return false
    }
}

// FP_SKIP_CAMERA_CHECK is only for build_app.py's tests.
let skipCameraCheck =
    ProcessInfo.processInfo.environment["FP_SKIP_CAMERA_CHECK"] != nil
if !skipCameraCheck && !cameraAllowed() {
    let choice = showAlert(
        "Filter Portal needs the camera",
        "Turn on Filter Portal in System Settings > Privacy & Security > "
        + "Camera, then open Filter Portal again.",
        ["Open Camera Settings", "Cancel"])
    if choice == .alertFirstButtonReturn, let url = URL(string:
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera") {
        NSWorkspace.shared.open(url)
    }
    exit(1)
}

guard let script = runScript else {
    showAlert("Filter Portal is damaged",
              "Run build_app.py again to rebuild the app.")
    exit(1)
}
let task = Process()
task.executableURL = URL(fileURLWithPath: "/bin/bash")
task.arguments = [script]
do {
    try task.run()
} catch {
    showAlert("Filter Portal couldn't start", error.localizedDescription)
    exit(1)
}
task.waitUntilExit()
exit(task.terminationStatus)
"""

# --------------------------------------------------------------------------
# run.sh: sets up Python if needed and runs filter_portal.py.
# @PROJECT@ and @BASE_PYTHON@ are filled in when the app is built.
# FP_PROJECT / FP_LOG / FP_NO_ALERT are only for build_app.py's own tests.
# --------------------------------------------------------------------------

RUN_SH = r"""#!/bin/bash
# Started by the Filter Portal app (made by build_app.py).
DEFAULT_PROJECT=@PROJECT@
PROJECT="${FP_PROJECT:-$DEFAULT_PROJECT}"
BASE_PYTHON=@BASE_PYTHON@
LOG="${FP_LOG:-$HOME/Library/Logs/Filter Portal.log}"

alert() {
  if [ -n "$FP_NO_ALERT" ]; then echo "ALERT: $1" >&2; return; fi
  /usr/bin/osascript - "$1" >/dev/null 2>&1 <<'END'
on run argv
  display alert "Filter Portal" message (item 1 of argv)
end run
END
}

notify() {
  [ -n "$FP_NO_ALERT" ] && return
  /usr/bin/osascript -e "display notification \"$1\" with title \"Filter Portal\"" >/dev/null 2>&1
}

mkdir -p "$(dirname "$LOG")"
echo "=== $(date) ===" >> "$LOG"

# Only one copy at a time (two would fight over the camera).
if pgrep -f "filter_portal.py" >/dev/null 2>&1; then
  alert "Filter Portal is already open. Its window may be hidden behind other windows."
  exit 0
fi

if [ ! -f "$PROJECT/filter_portal.py" ]; then
  alert "Can't find Filter Portal's files in:
$PROJECT

If you moved or renamed that folder, run build_app.py from its new place."
  exit 1
fi
cd "$PROJECT" || exit 1

# First run, or the .venv folder is missing: set up the Python packages.
if [ ! -x .venv/bin/python ]; then
  notify "Setting up for the first time. This can take a few minutes."
  if ! "$BASE_PYTHON" -m venv .venv >> "$LOG" 2>&1 \
     || ! .venv/bin/python -m pip install -r requirements.txt >> "$LOG" 2>&1; then
    rm -rf .venv   # half-finished setup: start over next time
    alert "Setting up Filter Portal failed. Is the internet on? Try again in a moment.

Details: ~/Library/Logs/Filter Portal.log"
    exit 1
  fi
fi

RUN_LOG="$(mktemp)"
.venv/bin/python filter_portal.py > "$RUN_LOG" 2>&1
status=$?
cat "$RUN_LOG" >> "$LOG"
if [ $status -ne 0 ]; then
  alert "Filter Portal stopped with an error:

$(grep -v -e '^INFO:' -e '^W0000' -e '^I0000' -e '^OpenCV:' -e 'WARN' "$RUN_LOG" | tail -n 4)

Details: ~/Library/Logs/Filter Portal.log"
fi
rm -f "$RUN_LOG"
exit $status
"""


# --------------------------------------------------------------------------
# Icon
# --------------------------------------------------------------------------

def draw_icon(size=1024):
    """The app icon: a purple rounded square with a white portal outline."""
    big = size * 2                      # draw at 2x, then shrink = smooth
    s = big / 1024                      # design coordinates are 0..1024
    icon = np.zeros((big, big, 4), np.uint8)

    # Rounded-square background with a top-to-bottom purple gradient.
    mask = np.zeros((big, big), np.uint8)
    inset, radius = int(100 * s), int(185 * s)
    lo, hi = inset, big - inset
    cv2.rectangle(mask, (lo + radius, lo), (hi - radius, hi), 255, -1)
    cv2.rectangle(mask, (lo, lo + radius), (hi, hi - radius), 255, -1)
    for cx in (lo + radius, hi - radius):
        for cy in (lo + radius, hi - radius):
            cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    t = np.linspace(0, 1, big)[:, None, None]
    top, bottom = np.array([95, 25, 45]), np.array([170, 60, 175])  # BGR
    icon[..., :3] = (top * (1 - t) + bottom * t).astype(np.uint8)
    icon[..., 3] = mask

    # The portal: a slightly skewed quad, lighter inside, white outline,
    # with a dot at each fingertip corner.
    quad = (np.array([[300, 330], [735, 285], [760, 705], [265, 735]])
            * s).astype(np.int32)
    inside = np.zeros((big, big), np.uint8)
    cv2.fillPoly(inside, [quad], 255, cv2.LINE_AA)
    glow = inside[..., None] / 255 * 0.35
    icon[..., :3] = (icon[..., :3] * (1 - glow) + 255 * glow).astype(np.uint8)
    white = (255, 255, 255, 255)
    cv2.polylines(icon, [quad], True, white, int(16 * s), cv2.LINE_AA)
    for x, y in quad:
        cv2.circle(icon, (int(x), int(y)), int(30 * s), white, -1,
                   cv2.LINE_AA)
    icon[..., 3] = np.minimum(icon[..., 3], mask)   # keep rounded corners
    return cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)


def make_icns(path):
    """Save the icon in every size macOS wants, as one .icns file."""
    master = draw_icon(1024)
    with tempfile.TemporaryDirectory() as folder:
        iconset = os.path.join(folder, "AppIcon.iconset")
        os.makedirs(iconset)
        for points in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                pixels = points * scale
                name = f"icon_{points}x{points}{'@2x' if scale == 2 else ''}"
                cv2.imwrite(os.path.join(iconset, name + ".png"),
                            cv2.resize(master, (pixels, pixels),
                                       interpolation=cv2.INTER_AREA))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", path],
                       check=True)


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------

def compile_launcher(output):
    """Compile the Swift launcher into `output`."""
    arch = "arm64" if platform.machine() == "arm64" else "x86_64"
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "main.swift")
        with open(source, "w") as f:
            f.write(LAUNCHER_SWIFT)
        result = subprocess.run(
            ["xcrun", "swiftc", "-O", "-swift-version", "5",
             "-target", f"{arch}-apple-macos12.0", source, "-o", output],
            capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit("Compiling the launcher failed (are Xcode or its command "
                 "line tools installed?):\n" + result.stderr)


def build(project, app_path):
    """Build the app bundle at app_path that runs filter_portal.py in project."""
    base_python = getattr(sys, "_base_executable", sys.executable)

    # Only ever replace an app that this script made.
    if os.path.exists(app_path):
        info = os.path.join(app_path, "Contents", "Info.plist")
        with open(info, "rb") as f:
            if plistlib.load(f).get("CFBundleIdentifier") != BUNDLE_ID:
                sys.exit(f"{app_path} exists and isn't ours; not touching it.")
        shutil.rmtree(app_path)

    contents = os.path.join(app_path, "Contents")
    os.makedirs(os.path.join(contents, "MacOS"))
    os.makedirs(os.path.join(contents, "Resources"))

    compile_launcher(os.path.join(contents, "MacOS", EXECUTABLE))

    run_sh = os.path.join(contents, "Resources", "run.sh")
    with open(run_sh, "w") as f:
        f.write(RUN_SH.replace("@PROJECT@", shlex.quote(project))
                      .replace("@BASE_PYTHON@", shlex.quote(base_python)))
    os.chmod(run_sh, 0o755)

    with open(os.path.join(contents, "Info.plist"), "wb") as f:
        plistlib.dump({
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleIdentifier": BUNDLE_ID,
            "CFBundleExecutable": EXECUTABLE,
            "CFBundleIconFile": "AppIcon",
            "CFBundlePackageType": "APPL",
            "CFBundleVersion": "2.0",
            "CFBundleShortVersionString": "2.0",
            "LSMinimumSystemVersion": "12.0",
            "LSUIElement": True,   # the launcher itself has no Dock icon
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": CAMERA_USAGE,
        }, f)

    make_icns(os.path.join(contents, "Resources", "AppIcon.icns"))

    # Ad-hoc signature: gives macOS a stable identity for the camera
    # permission. (No hardened runtime, so no camera entitlement needed.)
    subprocess.run(["codesign", "--force", "--sign", "-", app_path],
                   check=True, capture_output=True)
    if os.path.exists(LSREGISTER):   # so Spotlight finds it right away
        subprocess.run([LSREGISTER, "-f", app_path], check=False)
    return app_path


def add_to_dock(app_path):
    """Add the app to the end of the Dock, unless it's already there."""
    current = subprocess.run(
        ["defaults", "read", "com.apple.dock", "persistent-apps"],
        capture_output=True, text=True).stdout
    url = pathlib.Path(app_path).as_uri() + "/"
    if url in current or app_path in current:
        return False
    tile = (
        "<dict><key>tile-data</key><dict>"
        "<key>file-data</key><dict>"
        f"<key>_CFURLString</key><string>{url}</string>"
        "<key>_CFURLStringType</key><integer>15</integer>"
        "</dict>"
        f"<key>file-label</key><string>{APP_NAME}</string>"
        "</dict><key>tile-type</key><string>file-tile</string></dict>")
    subprocess.run(["defaults", "write", "com.apple.dock", "persistent-apps",
                    "-array-add", tile], check=True)
    subprocess.run(["killall", "Dock"], check=False)   # Dock reloads itself
    return True


def main():
    parser = argparse.ArgumentParser(description="Build Filter Portal.app")
    parser.add_argument("project", nargs="?", default=HERE,
                        help="folder with filter_portal.py (default: here)")
    parser.add_argument("--app-path", default=DEFAULT_APP_PATH)
    parser.add_argument("--no-dock", action="store_true",
                        help="don't add the app to the Dock")
    args = parser.parse_args()

    project = os.path.abspath(args.project)
    app_path = build(project, os.path.abspath(args.app_path))
    print(f"Built {app_path}")
    print(f"  runs: {os.path.join(project, 'filter_portal.py')}")
    if not args.no_dock:
        print("  added to the Dock" if add_to_dock(app_path)
              else "  already in the Dock")
    print("Open it from the Dock, or Cmd+Space and type 'Filter Portal'.")


if __name__ == "__main__":
    main()
