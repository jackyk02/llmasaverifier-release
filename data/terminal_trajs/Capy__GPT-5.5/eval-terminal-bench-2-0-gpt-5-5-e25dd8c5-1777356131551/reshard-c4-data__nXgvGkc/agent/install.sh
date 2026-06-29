#!/usr/bin/env bash
set -euo pipefail

SETUP_LOG_FILE="${CAPY_AGENT_SETUP_LOG_FILE:-/tmp/capy-agent-setup.log}"
mkdir -p "$(dirname "$SETUP_LOG_FILE")"
touch "$SETUP_LOG_FILE"
exec > >(tee -a "$SETUP_LOG_FILE") 2>&1

echo "[capy-agent-install] starting setup"

missing_packages=()
needs_ca_refresh=false
apt_indexes_refreshed=false

add_missing_package() {
  local pkg="$1"
  local existing
  for existing in "${missing_packages[@]}"; do
    if [ "$existing" = "$pkg" ]; then
      return
    fi
  done
  missing_packages+=("$pkg")
  if [ "$pkg" = "ca-certificates" ]; then
    needs_ca_refresh=true
  fi
}

retry_cmd() {
  local max_attempts="$1"
  local base_delay_seconds="$2"
  shift 2

  local attempt=1
  local exit_code=0

  while true; do
    if "$@"; then
      return 0
    fi

    exit_code=$?
    if [ "$attempt" -ge "$max_attempts" ]; then
      return "$exit_code"
    fi

    echo "[capy-agent-install] command failed (attempt ${attempt}/${max_attempts}), retrying: $*"
    sleep $((base_delay_seconds * attempt))
    attempt=$((attempt + 1))
  done
}

refresh_apt_indexes() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi

  if [ "$apt_indexes_refreshed" = true ]; then
    return 0
  fi

  echo "[capy-agent-install] refreshing apt package indexes"
  retry_cmd 3 2 apt-get update -o Acquire::Retries=3
  apt_indexes_refreshed=true
}

download_file() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    retry_cmd 3 2 curl --fail --show-error --location "$url" --output "$output"
    return 0
  fi

  if command -v wget >/dev/null 2>&1; then
    retry_cmd 3 2 wget -O "$output" "$url"
    return 0
  fi

  return 1
}

install_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    refresh_apt_indexes
    if ! retry_cmd 2 2 apt-get install -y --no-install-recommends -o Acquire::Retries=3 "$@"; then
      echo "[capy-agent-install] apt install failed after refresh; retrying with --fix-missing"
      apt_indexes_refreshed=false
      refresh_apt_indexes
      retry_cmd 3 2 apt-get install -y --no-install-recommends --fix-missing -o Acquire::Retries=3 "$@"
    fi
  elif command -v apk >/dev/null 2>&1; then
    retry_cmd 3 2 apk add --no-cache "$@"
  elif command -v dnf >/dev/null 2>&1; then
    retry_cmd 3 2 dnf install -y "$@"
  elif command -v yum >/dev/null 2>&1; then
    retry_cmd 3 2 yum install -y "$@"
  else
    echo "Missing required packages ($*), and no supported package manager was found"
    exit 1
  fi
}

ensure_command_available() {
  local command_name="$1"
  shift
  local packages=("$@")

  if command -v "$command_name" >/dev/null 2>&1; then
    return 0
  fi

  if [ "$#" -eq 0 ]; then
    echo "Required command '$command_name' is missing and no package mapping was provided" >&2
    return 1
  fi

  echo "[capy-agent-install] '$command_name' missing after install, retrying packages: ${packages[*]}"
  install_packages "${packages[@]}"

  if command -v "$command_name" >/dev/null 2>&1; then
    return 0
  fi

  echo "Required command '$command_name' is still unavailable after retries" >&2
  return 1
}

# Bun installer requires unzip and curl.
if ! command -v unzip >/dev/null 2>&1; then
  add_missing_package unzip
fi

if ! command -v curl >/dev/null 2>&1; then
  add_missing_package curl
fi

if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  add_missing_package wget
fi

if ! command -v rg >/dev/null 2>&1; then
  add_missing_package ripgrep
fi

# Ensure CA bundle package is present for reliable HTTPS downloads.
if command -v apt-get >/dev/null 2>&1; then
  if ! dpkg-query -W -f='${Status}' ca-certificates 2>/dev/null \
    | grep -q "install ok installed"; then
    add_missing_package ca-certificates
  fi
elif command -v apk >/dev/null 2>&1; then
  if ! apk info -e ca-certificates >/dev/null 2>&1; then
    add_missing_package ca-certificates
  fi
elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
  if ! rpm -q ca-certificates >/dev/null 2>&1; then
    add_missing_package ca-certificates
  fi
fi

# Some base images have curl but no CA bundle, which breaks HTTPS downloads.
if [ ! -s "/etc/ssl/certs/ca-certificates.crt" ] && [ ! -s "/etc/ssl/cert.pem" ]; then
  add_missing_package ca-certificates
fi

if [ -n "${missing_packages[*]}" ]; then
  echo "[capy-agent-install] installing missing packages: ${missing_packages[*]}"
  install_packages "${missing_packages[@]}"
fi

ensure_command_available unzip unzip
ensure_command_available rg ripgrep
ensure_command_available curl curl

if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  ensure_command_available curl curl
  if ! command -v wget >/dev/null 2>&1; then
    install_packages wget || true
  fi
fi

# Refresh the trust store if available.
if [ "$needs_ca_refresh" = true ] && command -v update-ca-certificates >/dev/null 2>&1; then
  update-ca-certificates || true
fi

# Some distros expose certs at /etc/ssl/cert.pem instead of the path curl expects.
if [ ! -s "/etc/ssl/certs/ca-certificates.crt" ] && [ -s "/etc/ssl/cert.pem" ]; then
  mkdir -p /etc/ssl/certs
  ln -sf /etc/ssl/cert.pem /etc/ssl/certs/ca-certificates.crt
fi

# Prefer explicit CA paths for curl in minimal images.
if [ -s "/etc/ssl/certs/ca-certificates.crt" ]; then
  export SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
  export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
elif [ -s "/etc/ssl/cert.pem" ]; then
  export SSL_CERT_FILE="/etc/ssl/cert.pem"
  export CURL_CA_BUNDLE="/etc/ssl/cert.pem"
fi

if ! command -v bun >/dev/null 2>&1; then
  echo "[capy-agent-install] installing bun"

  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    install_packages curl wget ca-certificates
  fi

  ensure_command_available unzip unzip

  attempt_bun_install() {
    local installer_path="/tmp/capy-install-bun.sh"
    if ! download_file "https://bun.sh/install" "$installer_path"; then
      return 1
    fi
    if ! retry_cmd 3 2 bash "$installer_path"; then
      rm -f "$installer_path"
      return 1
    fi
    rm -f "$installer_path"
    return 0
  }

  if ! attempt_bun_install; then
    echo "[capy-agent-install] initial bun install attempt failed; retrying after dependency recovery"
    if command -v apt-get >/dev/null 2>&1; then
      apt_indexes_refreshed=false
      refresh_apt_indexes
    fi
    install_packages unzip curl wget ca-certificates || true
    ensure_command_available unzip unzip
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
      echo "Unable to download Bun installer: neither curl nor wget is available" >&2
      exit 1
    fi
    if ! attempt_bun_install; then
      echo "Unable to install Bun after dependency recovery attempts" >&2
      exit 1
    fi
  fi
fi

# Expose bun on a stable non-login-shell path for Harbor exec commands.
BUN_BIN="$(command -v bun || true)"
if [ -z "$BUN_BIN" ]; then
  if [ -x "$HOME/.bun/bin/bun" ]; then
    BUN_BIN="$HOME/.bun/bin/bun"
  elif [ -x "/root/.bun/bin/bun" ]; then
    BUN_BIN="/root/.bun/bin/bun"
  fi
fi

if [ -z "$BUN_BIN" ] || [ ! -x "$BUN_BIN" ]; then
  echo "bun installation succeeded but binary was not found" >&2
  exit 1
fi

mkdir -p /usr/local/bin
ln -sf "$BUN_BIN" /usr/local/bin/bun
echo "[capy-agent-install] setup complete"