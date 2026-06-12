#!/usr/bin/env bash
# Install a built executable, replacing a possibly-running target safely.
# On Linux, mv over an executing binary succeeds (old inode stays mapped).
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <source> <dest>" >&2
  exit 1
fi

src=$1
dst=$2
tmp="${dst}.new.$$"

cp "$src" "$tmp"
chmod a+x "$tmp"
mv -f "$tmp" "$dst"
