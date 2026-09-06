"""Explicit acceptance policies for RIR diagnostics.

Recording quality, repeat consistency and full-response fidelity are separate
claims. A deliberately short early RIR need not reconstruct the room tail.
"""
from __future__ import annotations

import math

from .config import RepeatConfig


def reconstruction_issues(metrics: dict | None, config: RepeatConfig) -> list[str]:
    """Apply absolute reconstruction requirements only when explicitly selected."""
    if config.reconstruction_policy == "diagnostic":
        return []
    if metrics is None:
        return ["缺少完整响应重构指标"]
    correlation = metrics.get("minimum_correlation")
    nmse = metrics.get("worst_nmse_db")
    if any(value is None or not math.isfinite(value) for value in (correlation, nmse)):
        return ["完整响应重构指标不是有限值"]
    issues = []
    if correlation < config.minimum_reconstruction_correlation:
        issues.append(
            f"完整响应重构相关性 {correlation:.4f} 低于 "
            f"{config.minimum_reconstruction_correlation:.4f}"
        )
    if nmse > config.maximum_reconstruction_nmse_db:
        issues.append(
            f"完整响应重构 NMSE {nmse:.2f} dB 超过 "
            f"{config.maximum_reconstruction_nmse_db:.2f} dB"
        )
    return issues


def timing_issues(timing: dict) -> list[str]:
    """Do not certify stable but invalid GCC estimates as microphone delays."""
    issues = []
    reference = timing.get("reference_microphone_channel", 1)
    for channel in timing.get("per_channel", []):
        if channel["microphone_channel"] == reference:
            continue
        gcc = channel.get("gcc_phat", {})
        value = channel.get("gcc_phat_delay_samples")
        if (
            gcc.get("valid") is False
            or gcc.get("reliable") is False
            or value is None
            or not math.isfinite(value)
        ):
            issues.append(
                f"麦克风 {channel['microphone_channel']} 的 GCC-PHAT 延迟无效或不可靠"
            )
    return issues


def quality_summary(
    issues: list[str], warnings: list[str], config: RepeatConfig
) -> dict:
    scope = (
        "recording_repeat_and_full_response_reconstruction"
        if config.reconstruction_policy == "full_response"
        else "recording_and_repeat_consistency_only"
    )
    return {
        "status": "pass" if not issues else "review_required",
        "recommended_for_training": not issues,
        "training_recommendation_scope": scope,
        "reconstruction_policy": config.reconstruction_policy,
        "issues": list(dict.fromkeys(issues)),
        "warnings": list(dict.fromkeys(warnings)),
    }
