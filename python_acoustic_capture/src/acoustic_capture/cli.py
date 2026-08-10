"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

from .audio import check_hardware_settings, create_backend, format_hardware_status, list_devices
from .check import capture_input_check, capture_silent_duplex_check
from .config import load_config
from .demo import generate_demo_audio
from .general import capture_general_io
from .experiment import compile_rir_dataset, expand_experiment_plan
from .rir import capture_rir
from .scene import capture_scene_block


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acoustic-capture", description="多通道声学采集工具")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices", help="列出 PortAudio/ASIO 音频设备")
    demo = sub.add_parser("demo-audio", help="生成无版权的流程验收 WAV 文件")
    demo.add_argument("--output-dir", default="audio")
    demo.add_argument("--sample-rate", type=int, default=48_000)
    demo.add_argument("--duration", type=float, default=4.0)
    validate = sub.add_parser("validate", help="检查 YAML 配置")
    validate.add_argument("config")
    check = sub.add_parser("check-input", help="进行一次不播放的短时麦克风录制检查")
    check.add_argument("config")
    check.add_argument("--duration", type=float, default=5.0)
    duplex = sub.add_parser("check-duplex", help="播放数字静音并检查同步双工音频流")
    duplex.add_argument("config")
    duplex.add_argument("--duration", type=float, default=1.0)
    hardware = sub.add_parser("hardware-check", help="检查真实声卡的设备、通道和采样率")
    hardware.add_argument("config")
    io = sub.add_parser("io", help="执行仅播放、仅录制或同步播录")
    io.add_argument("config")
    io.add_argument("--action", choices=["play", "record", "play_record"])
    rir = sub.add_parser("rir", help="重复采集、质检并平均 ESS 脉冲响应")
    rir.add_argument("config")
    rir.add_argument("--output-channel", type=int, help="临时指定扫频输出通道")
    scene = sub.add_parser("scene", help="采集可选的语音增强场景块")
    scene.add_argument("config")
    expand = sub.add_parser("plan-expand", help="把实验计划展开成逐条件 YAML")
    expand.add_argument("plan")
    dataset = sub.add_parser("rir-dataset", help="汇总已完成 RIR 并生成服务器训练索引")
    dataset.add_argument("plan")
    gui = sub.add_parser("gui", help="打开图形配置工具")
    gui.add_argument("config", nargs="?", help="要打开的 YAML 配置（可选）")
    gui.add_argument("--run-once", choices=["io", "rir", "scene"], help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "devices":
            print(list_devices())
        elif args.command == "demo-audio":
            files = generate_demo_audio(args.output_dir, args.sample_rate, args.duration)
            print(json.dumps({name: str(path) for name, path in files.items()}, ensure_ascii=False, indent=2))
        elif args.command == "validate":
            config = load_config(args.config)
            print(f"配置有效：{args.config}\n音频后端：{config.audio.backend}")
        elif args.command == "check-input":
            config = load_config(args.config)
            store = capture_input_check(config, create_backend(config.audio), args.duration)
            print(store.root)
            warnings = store.manifest.get("summary", {}).get("warnings", [])
            if warnings:
                print("质量警告：" + "；".join(warnings), file=sys.stderr)
                return 2
        elif args.command == "hardware-check":
            config = load_config(args.config)
            print(format_hardware_status(check_hardware_settings(config.audio)))
        elif args.command == "check-duplex":
            config = load_config(args.config)
            store = capture_silent_duplex_check(config, create_backend(config.audio), args.duration)
            print(store.root)
            warnings = store.manifest.get("summary", {}).get("warnings", [])
            if warnings:
                print("质量警告：" + "；".join(warnings), file=sys.stderr)
                return 2
        elif args.command == "io":
            config = load_config(args.config)
            if args.action:
                config.general.action = args.action
            store = capture_general_io(config, create_backend(config.audio))
            print(store.root)
        elif args.command == "rir":
            config = load_config(args.config)
            store = capture_rir(config, create_backend(config.audio), args.output_channel)
            print(store.root)
        elif args.command == "scene":
            config = load_config(args.config)
            store = capture_scene_block(config, create_backend(config.audio))
            print(store.root)
        elif args.command == "plan-expand":
            for path in expand_experiment_plan(args.plan):
                print(path)
        elif args.command == "rir-dataset":
            print(compile_rir_dataset(args.plan))
        elif args.command == "gui":
            from .gui import main as gui_main

            gui_main(args.config, run_once=args.run_once)
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
