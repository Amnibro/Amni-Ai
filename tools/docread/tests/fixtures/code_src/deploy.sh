#!/bin/bash
set -e

log() {
  echo "[deploy] $1"
}

function build_all {
  log "building"
  make all
}

build_all
