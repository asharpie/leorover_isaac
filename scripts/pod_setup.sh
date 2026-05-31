#!/bin/bash
# pod_setup.sh — persistent boot script for RunPod
#
# Lives on /workspace/pod_setup.sh (copied there once, then run by the
# container's start command on every boot). Restores SSH host keys,
# authorized_keys, and bashrc tweaks from the persistent volume so they
# survive Stop/Start cycles (which wipe the container layer).
#
# To install on a pod:
#   1. Save this file as /workspace/pod_setup.sh
#   2. Run it once interactively to populate /workspace/ssh/ from
#      whatever's currently in /etc/ssh/ and /root/.ssh/
#      (`bash /workspace/pod_setup.sh --init`)
#   3. Change container start command in RunPod dashboard to:
#      bash -c "bash /workspace/pod_setup.sh; sleep infinity"
#   4. Save as custom template so the script's existence on /workspace
#      and the start command are part of the template
#
# After that, every Stop/Start of any pod from that template just works.

set -e

WORKSPACE_SSH_DIR="/workspace/ssh"
HOST_KEYS_DIR="${WORKSPACE_SSH_DIR}/host_keys"
AUTHORIZED_KEYS_SRC="${WORKSPACE_SSH_DIR}/authorized_keys"

mkdir -p "${WORKSPACE_SSH_DIR}" "${HOST_KEYS_DIR}"

# ─── --init flag: capture current state into /workspace ───────────────────
# Use this once interactively after a manual SSH setup, so the persistent
# volume gets a snapshot of working host keys + authorized_keys.
if [ "$1" = "--init" ]; then
    echo "[pod_setup] Initializing persistent SSH state on /workspace..."

    if ls /etc/ssh/ssh_host_*_key 1>/dev/null 2>&1; then
        cp /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub "${HOST_KEYS_DIR}/"
        echo "[pod_setup]  - saved host keys to ${HOST_KEYS_DIR}"
    else
        echo "[pod_setup]  - WARNING: no host keys found in /etc/ssh/; run 'ssh-keygen -A' first"
    fi

    if [ -f /root/.ssh/authorized_keys ]; then
        cp /root/.ssh/authorized_keys "${AUTHORIZED_KEYS_SRC}"
        echo "[pod_setup]  - saved authorized_keys to ${AUTHORIZED_KEYS_SRC}"
    else
        echo "[pod_setup]  - WARNING: no /root/.ssh/authorized_keys to save"
    fi

    echo "[pod_setup] Done. From here on, bash /workspace/pod_setup.sh (no flag) will restore these on boot."
    exit 0
fi

# ─── normal boot: restore SSH state ───────────────────────────────────────

# Restore host keys from persistent storage (or generate fresh if missing)
if ls "${HOST_KEYS_DIR}"/ssh_host_*_key 1>/dev/null 2>&1; then
    mkdir -p /etc/ssh
    cp "${HOST_KEYS_DIR}"/ssh_host_*_key /etc/ssh/
    cp "${HOST_KEYS_DIR}"/ssh_host_*_key.pub /etc/ssh/ 2>/dev/null || true
    chmod 600 /etc/ssh/ssh_host_*_key
    chmod 644 /etc/ssh/ssh_host_*_key.pub 2>/dev/null || true
    echo "[pod_setup] Restored SSH host keys from ${HOST_KEYS_DIR}"
else
    echo "[pod_setup] No saved host keys found, generating fresh..."
    ssh-keygen -A
    cp /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub "${HOST_KEYS_DIR}/"
    echo "[pod_setup] Saved newly-generated host keys to ${HOST_KEYS_DIR} for next boot"
fi

# Restore authorized_keys from persistent storage
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ -f "${AUTHORIZED_KEYS_SRC}" ]; then
    cp "${AUTHORIZED_KEYS_SRC}" /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "[pod_setup] Restored authorized_keys from ${AUTHORIZED_KEYS_SRC}"
else
    echo "[pod_setup] WARNING: no ${AUTHORIZED_KEYS_SRC} to restore — SSH will refuse logins until you add it"
fi

# Auto-cd into IsaacLab on shell login (re-add each boot since bashrc is wiped)
if [ -d /workspace/IsaacLab ] && ! grep -q "cd /workspace/IsaacLab" /root/.bashrc 2>/dev/null; then
    echo "cd /workspace/IsaacLab" >> /root/.bashrc
fi

# Start SSH daemon
service ssh start || /usr/sbin/sshd

echo "[pod_setup] SSH ready. Pod is up."
