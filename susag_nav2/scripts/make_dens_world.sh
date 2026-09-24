#!/usr/bin/env bash
# Create a dynamic-obstacle world with N obstacles (2026-09-21).
#   ~/robohouse_ws/make_dens_world.sh <N> [seed=1] [radius_scale=1.0]
# Writes world_d<N>_<seed>.world (or world_d<N>_<seed>_r<pct>.world when radius_scale != 1),
# then launch Gazebo/Nav2 with  world_idx:=<that name>.
# Same generator settings as dyn1..dyn5, so with the same seed the first 10 obstacles follow
# the same tracks as dyn<seed>; only obstacles 11..N are added. radius_scale only shrinks the
# cylinders (0.25/0.20 m x scale); the tracks are unchanged.
# The controller listens to obstacle topics 1..20 (navigation_tgmppi_tight.yaml), so N > 20
# needs that list extended first.
set -e
N=${1:?usage: make_dens_world.sh <N> [seed] [radius_scale]}
SEED=${2:-1}
SCALE=${3:-1.0}
if [ "$N" -gt 20 ]; then
  echo "WARNING: N=$N > 20: the controller only subscribes to obstacle topics 1..20; obstacles beyond 20 would be invisible to it." >&2
fi
NAME="d${N}_${SEED}"
if [ "$SCALE" != "1.0" ] && [ "$SCALE" != "1" ]; then
  NAME="${NAME}_r$(python3 -c "print(round($SCALE*100))")"
fi
cd "$HOME/robohouse_ws/src/BARN_dataset/scaled_1/world_files"
python3 make_open_world.py --motion polynomial --obstacles "$N" --vx-max 1.5 --seed "$SEED" \
  --radius-scale "$SCALE" --scenario "$NAME" | grep -E "^wrote|problem|could not|launch with"
