"""S4 live-speaker capture. Repeated readings are explicitly NOT paired targets."""
from pathlib import Path
import csv
import time
import numpy as np
from .signals import load_audio, fit_duration, scale_dbfs, route_outputs
from .storage import RunStore
from .live_session import capture_session
from .quality import channel_metrics, multichannel_health_metrics

DEFAULT_CORPUS = """从这到百货大楼有多远？十三四公里。怎么坐车？过马路先坐二路公共汽车，然后换地铁。他是一个略显瘦小的男孩。
主楼位于校园的东北侧
他的病情得到了很好控制
人们再也不能这样等下去了
他为儿子买了一整根甘蔗
市区的停车收费将大幅提高
她醒来后发现自己脸上有黑眼圈
大风刮倒了一处在建厂房
姚大爷觉得车夫的想法蛮有道理
汹涌的河水顺流而下流得很快
坚持终于让他有所收获
据说这是当地最古老的小区
你就是那个爱打篮球的人
总理对任何事情都要刨根问底
渐渐地她还真就睡着了
这身衣服就像被大雨淋过似的
仓库的后面是一间小屋
太阳从东方升起来
哈尔滨在中国的最北面"""


def read_corpus(path=""):
    text = Path(path).read_text(encoding="utf-8-sig") if path else DEFAULT_CORPUS
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("朗读语料不能为空，请选择 UTF-8 文本，每行一句")
    return lines


@capture_session
def capture_human(config, backend, log=print, stop_requested=None, progress=None):
    stop = stop_requested or (lambda: False)
    emit = progress or (lambda update: None)
    corpus = read_corpus(config.human.corpus_file)
    folder = Path(config.scene.interferer_folder)
    sources = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"})
    if not sources:
        raise ValueError("请选择包含 WAV 或 FLAC 音频的干扰文件夹")
    rng = np.random.default_rng(config.scene.pairing_seed)
    rng.shuffle(sources)
    fs = config.audio.sample_rate
    frames = round(config.human.duration_s * fs)
    store = RunStore.create(config, "human")
    rows = []
    warning = "真人 target-only 与 mixed 是独立朗读，非严格配对监督参考；朗读提示不是已核验转写。"
    summary = {"warnings": [warning], "strict_supervised_pair": False, "completed_stages": 0}
    store.write_json("references/reading_corpus.json", {"lines": corpus})
    fields = ["round", "stage", "recording", "clip", "start_sample", "end_sample", "prompt_text", "interferer_source", "strict_supervised_pair", "status", "quality_passed", "timing_discontinuity", "quality_issues"]

    def save_labels():
        path = store.root / "raw" / "labels.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        store.add_artifact("raw/labels.csv")
        store.checkpoint()

    save_labels()
    try:
        for ordinal in range(config.human.rounds):
            source_path = sources[ordinal % len(sources)]
            noise = scale_dbfs(fit_duration(load_audio(source_path, fs), frames), config.scene.interferer_level_dbfs)
            # Soft edges prevent clicks at repeated playback boundaries.
            fade = min(round(.02 * fs), len(noise) // 2)
            if fade:
                noise[:fade] *= np.linspace(0, 1, fade)
                noise[-fade:] *= np.linspace(1, 0, fade)
            # One corpus line per round; keep it fixed across all three stages.
            prompt = corpus[ordinal % len(corpus)]
            for stage in ("target_only", "interferer_only", "mixture"):
                if stop():
                    break
                emit({"event": "human_prepared", "stage": stage, "prompt": prompt,
                      "round": ordinal + 1, "rounds": config.human.rounds, "frames": frames,
                      "sample_rate": fs, "preparation_s": config.human.preparation_s})
                deadline = time.monotonic() + config.human.preparation_s
                while time.monotonic() < deadline and not stop():
                    time.sleep(.03)
                if stop():
                    break
                log(f"S4 {ordinal + 1}/{config.human.rounds}: {stage}")
                if stage == "target_only":
                    result = backend.record(frames)
                else:
                    output = route_outputs({config.audio.interferer_output_channel: noise}, frames)
                    result = backend.play_record(output)
                raw = f"raw/round_{ordinal + 1:03d}_{stage}.wav"
                store.write_audio(raw, result.microphones, fs)
                channels = channel_metrics(result.microphones)
                health = multichannel_health_metrics(result.microphones)
                issues = []
                if any(item["clipped"] for item in channels):
                    issues.append("削波")
                if health["has_nonfinite_samples"]:
                    issues.append("非有限音频样点")
                if health["has_silent_channel"]:
                    issues.append("静音通道")
                if health["exact_duplicate_channel_pairs"]:
                    issues.append("录制通道完全重复")
                store.write_json(f"metrics/round_{ordinal + 1:03d}_{stage}.json", {
                    **result.status, "channels": channels, "array_health": health,
                    "quality_issues": issues,
                })
                actual = len(result.microphones)
                bad = bool(result.status.get("xrun") or result.status.get("timing_discontinuity"))
                status = "timing_discontinuity" if result.status.get("timing_discontinuity") else "xrun" if bad else "partial" if actual < frames else "complete"
                base = dict(round=ordinal + 1, stage=stage, recording=raw, prompt_text=prompt if stage != "interferer_only" else "",
                            interferer_source=str(source_path) if stage != "target_only" else "",
                            strict_supervised_pair=False, status=status, quality_passed=status == "complete" and not issues,
                            quality_issues="; ".join(issues),
                            timing_discontinuity=bool(result.status.get("timing_discontinuity")))
                # Always index raw audio, including a cancelled stage shorter than 4 s.
                rows.append(dict(base, clip="", start_sample=0, end_sample=actual))
                chunk = 4 * fs
                for begin in range(0, actual - chunk + 1, chunk):
                    clip = f"processed/round_{ordinal + 1:03d}_{stage}_{begin // chunk + 1:04d}.wav"
                    store.write_audio(clip, result.microphones[begin:begin + chunk], fs)
                    rows.append(dict(base, clip=clip, start_sample=begin, end_sample=begin + chunk))
                if bad:
                    summary["warnings"].append(f"第 {ordinal + 1} 轮 {stage} 发生音频丢帧，见 labels.csv")
                if issues:
                    summary["warnings"].append(f"第 {ordinal + 1} 轮 {stage}：{'、'.join(issues)}，不合格录音已保留")
                if status == "complete":
                    summary["completed_stages"] += 1
                save_labels()
                if actual < frames and not stop():
                    raise RuntimeError("真人录制提前结束，已保留原始录音和标签，请检查声卡")
            if stop():
                break
        store.finish(summary, status="cancelled" if stop() else "completed")
        return store
    except Exception as exc:
        save_labels()
        store.finish(dict(summary, error=str(exc)), status="failed")
        raise
