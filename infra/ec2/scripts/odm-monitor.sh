#!/bin/bash
# Telemetry stack installer: node_exporter + cAdvisor + Grafana Alloy.
# Pushes metrics and logs to Grafana Cloud; posts stage annotations via REST API.
# Sources /etc/odm-env for GRAFANA_* credentials.
# Silent no-op when GRAFANA_API_KEY is empty — monitoring is fully optional.
set -euo pipefail

source /etc/odm-env

if [ -z "${GRAFANA_API_KEY:-}" ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: GRAFANA_API_KEY not set — skipping telemetry setup"
  exit 0
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: installing telemetry stack"

# ── Instance identity ──────────────────────────────────────────────────────────
INSTANCE_ID=$(curl -sf --max-time 3 \
  http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || hostname)
# Sanitise PROJECT for use as a label (replace / and spaces with _)
PROJECT_LABEL="${PROJECT//[\/\ ]/_}"

# ── node_exporter ──────────────────────────────────────────────────────────────
NODE_EXP_VERSION="1.8.2"
NODE_EXP_URL="https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXP_VERSION}/node_exporter-${NODE_EXP_VERSION}.linux-amd64.tar.gz"

if ! command -v node_exporter &>/dev/null; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: installing node_exporter ${NODE_EXP_VERSION}"
  curl -sL "${NODE_EXP_URL}" \
    | tar -xz --strip-components=1 -C /usr/local/bin/ \
        "node_exporter-${NODE_EXP_VERSION}.linux-amd64/node_exporter"
fi

TEXTFILE_DIR=/var/lib/node_exporter/textfile_collector
mkdir -p "${TEXTFILE_DIR}"
# node_exporter runs as nobody — ensure it can read the textfile dir
chmod 755 "${TEXTFILE_DIR}"

cat > /etc/systemd/system/node_exporter.service << EOF
[Unit]
Description=Prometheus Node Exporter
After=network.target

[Service]
User=nobody
ExecStart=/usr/local/bin/node_exporter \\
  --collector.systemd \\
  --collector.processes \\
  --collector.diskstats \\
  --collector.textfile.directory=${TEXTFILE_DIR} \\
  --collector.filesystem.mount-points-exclude="^/(dev|proc|sys|run)(\$|/)"
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now node_exporter
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: node_exporter started on :9100"

# ── cAdvisor (per-container metrics) ──────────────────────────────────────────
CADVISOR_VERSION="v0.49.2"

if ! docker ps --format '{{.Names}}' | grep -q '^cadvisor$'; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: starting cAdvisor ${CADVISOR_VERSION}"
  docker run -d --name cadvisor \
    --restart unless-stopped \
    -p 8080:8080 \
    --volume=/:/rootfs:ro \
    --volume=/var/run:/var/run:ro \
    --volume=/sys:/sys:ro \
    --volume=/var/lib/docker:/var/lib/docker:ro \
    --detach-keys="" \
    gcr.io/cadvisor/cadvisor:"${CADVISOR_VERSION}" \
    2>&1 | tail -1
fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: cAdvisor started on :8080"

# ── Grafana Alloy ──────────────────────────────────────────────────────────────
if ! command -v alloy &>/dev/null; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: installing Grafana Alloy"
  rpm --import https://rpm.grafana.com/gpg.key
  cat > /etc/yum.repos.d/grafana.repo << 'EOF'
[grafana]
name=grafana
baseurl=https://rpm.grafana.com
repo_gpgcheck=1
enabled=1
gpgcheck=1
gpgkey=https://rpm.grafana.com/gpg.key
sslverify=1
sslcacert=/etc/pki/tls/certs/ca-bundle.crt
EOF
  dnf install -y alloy
fi

# ── Alloy environment file (credentials) ──────────────────────────────────────
mkdir -p /etc/alloy
cat > /etc/alloy/alloy-env << ENVEOF
GRAFANA_PROM_URL=${GRAFANA_PROM_URL}
GRAFANA_PROM_USER=${GRAFANA_PROM_USER}
GRAFANA_LOKI_URL=${GRAFANA_LOKI_URL}
GRAFANA_LOKI_USER=${GRAFANA_LOKI_USER}
GRAFANA_API_KEY=${GRAFANA_API_KEY}
INSTANCE_ID=${INSTANCE_ID}
PROJECT=${PROJECT_LABEL}
ENVEOF
chmod 600 /etc/alloy/alloy-env

# ── Alloy config ───────────────────────────────────────────────────────────────
cat > /etc/alloy/config.alloy << 'EOF'
// ── node_exporter → Grafana Cloud ─────────────────────────────────────────────
prometheus.scrape "node" {
  targets = [{
    __address__ = "localhost:9100",
    job         = "node_exporter",
    instance    = env("INSTANCE_ID"),
    project     = env("PROJECT"),
  }]
  scrape_interval = "30s"
  scrape_timeout  = "25s"
  forward_to = [prometheus.remote_write.grafana_cloud.receiver]
}

// ── cAdvisor (per-ODM-container) → Grafana Cloud ──────────────────────────────
prometheus.scrape "cadvisor" {
  targets = [{
    __address__ = "localhost:8080",
    job         = "cadvisor",
    instance    = env("INSTANCE_ID"),
    project     = env("PROJECT"),
  }]
  scrape_interval = "30s"
  scrape_timeout  = "25s"
  forward_to = [prometheus.remote_write.grafana_cloud.receiver]
}

// ── remote_write endpoint ─────────────────────────────────────────────────────
prometheus.remote_write "grafana_cloud" {
  endpoint {
    url = env("GRAFANA_PROM_URL")
    basic_auth {
      username = env("GRAFANA_PROM_USER")
      password = env("GRAFANA_API_KEY")
    }
  }
}

// ── ODM bootstrap log → Loki ──────────────────────────────────────────────────
loki.source.file "odm_log" {
  targets = [{
    __path__  = "/var/log/odm-bootstrap.log",
    job       = "odm",
    instance  = env("INSTANCE_ID"),
    project   = env("PROJECT"),
  }]
  forward_to = [loki.write.grafana_cloud.receiver]
}

loki.write "grafana_cloud" {
  endpoint {
    url = env("GRAFANA_LOKI_URL")
    basic_auth {
      username = env("GRAFANA_LOKI_USER")
      password = env("GRAFANA_API_KEY")
    }
  }
}
EOF

# ── Systemd override: inject EnvironmentFile into alloy service ────────────────
mkdir -p /etc/systemd/system/alloy.service.d
cat > /etc/systemd/system/alloy.service.d/override.conf << 'EOF'
[Service]
EnvironmentFile=/etc/alloy/alloy-env
EOF

systemctl daemon-reload
systemctl enable --now alloy
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: Alloy started — pushing to Grafana Cloud"

# ── ODM progress reporter (cron, every minute) ─────────────────────────────────
aws s3 cp "s3://${BUCKET}/odm-scripts/odm-progress.sh" \
  /usr/local/bin/odm-progress.sh --region "${REGION}" 2>/dev/null || true
chmod +x /usr/local/bin/odm-progress.sh

# Install cron entry for root (needs docker access)
CRON_LINE="* * * * * /usr/local/bin/odm-progress.sh >> /var/log/odm-progress.log 2>&1"
( crontab -l 2>/dev/null | grep -v odm-progress.sh; echo "${CRON_LINE}" ) | crontab -
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: progress reporter cron installed (every 1min)"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-monitor: telemetry stack ready"
