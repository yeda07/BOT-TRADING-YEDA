from app.monitoring.system_monitor import SystemMonitor


def test_system_monitor_collects_without_psutil_requirement():
    metrics = SystemMonitor().collect()

    assert "disk_usage_percent" in metrics
    assert "process_uptime_seconds" in metrics
