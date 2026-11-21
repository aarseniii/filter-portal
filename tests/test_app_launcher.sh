#!/bin/bash
# End-to-end test of the Mac app's launcher (build_app.py), using FAKE
# projects. It builds a throwaway copy of the app in a temp folder (not
# touching your real app or Dock), never asks for the camera
# (FP_SKIP_CAMERA_CHECK) and never shows message boxes (FP_NO_ALERT).
#
# Run from the project folder:   bash tests/test_app_launcher.sh
cd "$(dirname "$0")/.." || exit 1
PROJECT_PYTHON=.venv/bin/python
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
APP="$W/Filter Portal.app"
BIN="$APP/Contents/MacOS/FilterPortal"
BASE="$("$PROJECT_PYTHON" -c 'import sys; print(sys._base_executable)')"

echo "Building a test copy of the app..."
"$PROJECT_PYTHON" build_app.py --no-dock --app-path "$APP" >/dev/null 2>&1 \
  || { echo "FAIL  build"; exit 1; }

export FP_SKIP_CAMERA_CHECK=1 FP_NO_ALERT=1 FP_LOG="$W/test.log"
pass=0; fail=0
check() {
  if eval "$2"; then echo "PASS  $1"; pass=$((pass+1))
  else echo "FAIL  $1"; fail=$((fail+1)); fi
}
make_project() {  # $1 = folder, $2 = fake filter_portal.py, $3 = venv yes/no
  mkdir -p "$1"; printf '%s\n' "$2" > "$1/filter_portal.py"
  : > "$1/requirements.txt"
  if [ "$3" = yes ]; then mkdir -p "$1/.venv/bin"; ln -s "$BASE" "$1/.venv/bin/python"; fi
}

if pgrep -f "filter_portal.py" >/dev/null; then
  echo "Filter Portal is running; quit it first (the launcher refuses a 2nd copy)."
  exit 1
fi

check "self-check runs" '"$BIN" --check >/dev/null'

make_project "$W/ok" 'print("FAKE APP RAN")' yes
FP_PROJECT="$W/ok" "$BIN" 2> "$W/a.err"; code=$?
check "normal run exits 0" '[ $code -eq 0 ]'
check "app output reaches the log" 'grep -q "FAKE APP RAN" "$W/test.log"'
check "no message box on success" '! grep -q "^ALERT:" "$W/a.err"'

make_project "$W/fail" 'import sys
print("OpenCV: not authorized to capture video (status 0), requesting...")
print("[ WARN:0@0.3] global cap_ffmpeg_impl.hpp:1222 noise")
sys.exit("Could not open the webcam. Allow camera access for Filter Portal.")' yes
FP_PROJECT="$W/fail" "$BIN" 2> "$W/b.err"; code=$?
check "failing app exits 1" '[ $code -eq 1 ]'
check "message box shows the real error" 'grep -q "Could not open the webcam" "$W/b.err"'
check "message box hides OpenCV noise" '! grep -q -e "OpenCV:" -e "WARN" "$W/b.err"'

FP_PROJECT="$W/nowhere" "$BIN" 2> "$W/c.err"; code=$?
check "missing folder exits 1" '[ $code -eq 1 ]'
check "missing folder is explained" 'grep -q "find Filter Portal" "$W/c.err"'

make_project "$W/fresh" 'print("FRESH RAN")' no
FP_PROJECT="$W/fresh" "$BIN" 2> "$W/d.err"; code=$?
check "first-run setup exits 0" '[ $code -eq 0 ]'
check "first-run setup creates .venv" '[ -x "$W/fresh/.venv/bin/python" ]'
check "app runs after setup" 'grep -q "FRESH RAN" "$W/test.log"'

make_project "$W/bad" 'print("SHOULD NOT RUN")' no
echo "this-package-does-not-exist-fp-test==0.0.1" > "$W/bad/requirements.txt"
FP_PROJECT="$W/bad" "$BIN" 2> "$W/e.err"; code=$?
check "failed setup exits 1" '[ $code -eq 1 ]'
check "failed setup is explained" 'grep -q "Setting up Filter Portal failed" "$W/e.err"'
check "half-made .venv is removed" '[ ! -e "$W/bad/.venv" ]'
check "app doesn't run after failed setup" '! grep -q "SHOULD NOT RUN" "$W/test.log"'

"$BASE" -c "import time; time.sleep(20)" filter_portal.py & dummy=$!; sleep 1
FP_PROJECT="$W/ok" "$BIN" 2> "$W/f.err"; code=$?
kill $dummy 2>/dev/null; wait $dummy 2>/dev/null
check "second copy exits 0" '[ $code -eq 0 ]'
check "second copy says already open" 'grep -q "already open" "$W/f.err"'

echo "---- $pass passed, $fail failed"
[ $fail -eq 0 ]
