#!/bin/bash

set -e

export TERM=xterm-256color

RESET=$(tput sgr0)
RED=$(tput setaf 9)
GREEN=$(tput setaf 10)
YELLOW=$(tput setaf 11)
BLUE=$(tput setaf 12)
PURPLE=$(tput setaf 13)
CYAN=$(tput setaf 14)

# from https://starship.rs/install.sh
info() {
  printf "%s\n" "${CYAN}> $*${RESET}"
}

warn() {
  printf "%s\n" "${YELLOW}! $*${RESET}"
}

error() {
  printf "%s\n" "${RED}x $*${RESET}"
}

success() {
  printf "%s\n" "${GREEN}✓ $*${RESET}"
}

CURRENT_DIR=$(pwd)
SUBMODULES="static/examples"

check_submodules() {
  # for module in ${SUBMODULES[@]}
  # do
  cd ${SUBMODULES}
  FROM_HASH=$(git rev-parse HEAD)

  cd ${CURRENT_DIR}
  git submodule update --remote ${SUBMODULES}

  cd ${SUBMODULES}
  TO_HASH=$(git rev-parse HEAD)

  CHANGED=$(git diff --name-only ${FROM_HASH}...${TO_HASH})

  if [ -z "$CHANGED" ]; then
    success "Submodule ${SUBMODULES} is up to date."
  else
    error "Submodule ${SUBMODULES} is not up to date."
    info "Run git submodule update --remote ${SUBMODULES} to update."
    exit 1
  fi

  cd ${CURRENT_DIR}
  # done
}

check_submodules
