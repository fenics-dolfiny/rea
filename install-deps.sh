#!/usr/bin/env sh
set -euo pipefail

apk add --no-cache \
    clang \
    clang-extra-tools \
    cmake \
    build-base \
    ninja \
    zsh \
    shadow
