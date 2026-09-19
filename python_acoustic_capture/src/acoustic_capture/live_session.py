"""One thread-affine stream per experiment; settle before any measurement.

Idle input is discarded, never written. Only an explicit measurement job is
retained. Keeping the stream open avoids re-triggering USB input startup gaps.
"""
from functools import wraps
import threading
import time
import numpy as np


def capture_session(function):
    @wraps(function)
    def wrapped(config, backend, *args, **kwargs):
        session = getattr(backend, 'experiment_session', None)
        if session is None:
            return function(config, backend, *args, **kwargs)
        if function.__name__ == 'capture_general_io':
            outputs = 0 if config.general.action == 'record' else config.general.output_channel
        elif function.__name__ == 'capture_rir':
            outputs = kwargs.get('output_channel') or (args[0] if args else None) or config.audio.target_output_channel
        elif function.__name__ == 'capture_human':
            outputs = config.audio.interferer_output_channel
        else:
            outputs = max(config.audio.target_output_channel, config.audio.interferer_output_channel)
        with session(outputs):
            return function(config, backend, *args, **kwargs)
    return wrapped


class LiveSession:
    def __init__(self, backend, outputs):
        self.backend = backend
        self.outputs = outputs
        self.stream = None
        self.job = None
        self.last_time = None
        self.stable_since = None
        self.ready = False
        self.error = None
        self.opened = time.monotonic()
        self.latency = None
        self.initial_settling_seconds = None

    def callback(self, indata, outdata, frames, time_info, flags):
        now = time.monotonic()
        period = frames / self.backend.config.sample_rate
        gap = 0 if self.last_time is None else now - self.last_time
        self.last_time = now
        discontinuity = gap > max(.15, 4 * period)
        if flags or discontinuity or self.stable_since is None:
            self.stable_since = now
        self.ready = now - self.stable_since >= 1.0
        if self.ready and self.initial_settling_seconds is None:
            self.initial_settling_seconds = now - self.opened
        if outdata is not None:
            outdata.fill(0)
        job = self.job
        if job is None or job['done'].is_set():
            return
        # A stale ready flag from before file/FFT work is not sufficient.
        # Activate at a clean callback boundary, not on the owner thread.
        if job['cursor'] == 0 and (not self.ready or flags):
            return
        status = job['status']
        if job['cursor'] == 0:
            status['preparation_seconds'] = now - job['requested_at']
        self.backend._merge_callback_status(status, flags)
        self.backend._record_callback_time(status, time_info)
        status['max_callback_gap_seconds'] = max(status['max_callback_gap_seconds'], gap)
        if discontinuity and job['cursor'] > 0:
            status['timing_discontinuity'] = True
        cursor = job['cursor']
        count = min(frames, len(job['recording']) - cursor)
        job['recording'][cursor:cursor + count] = indata[:count]
        if job['output'] is not None:
            outdata[:count, :job['output'].shape[1]] = job['output'][cursor:cursor + count]
        job['cursor'] += count
        if job['cursor'] >= len(job['recording']) or self.backend._abort_requested.is_set():
            job['done'].set()

    def open(self):
        self.opened = time.monotonic()
        b = self.backend
        sd = b._module()
        b.check_settings(input_required=True, output_channels=self.outputs or None,
                         validate_format=False)
        inputs, outputs = b._devices()
        def guarded(*args):
            try:
                if self.outputs:
                    self.callback(*args)
                else:
                    data, frames, ti, flags = args
                    self.callback(data, None, frames, ti, flags)
            except Exception as exc:
                self.error = exc
                if self.outputs:
                    args[1].fill(0)
                raise sd.CallbackAbort() from exc
        options = dict(samplerate=b.config.sample_rate, blocksize=b.config.block_size,
                       dtype=b.config.dtype, latency=b.config.latency, callback=guarded)
        if self.outputs:
            self.stream = sd.Stream(device=(b._device_argument(inputs), b._device_argument(outputs)),
                channels=(max(b.config.input_channels), self.outputs), **options)
        else:
            self.stream = sd.InputStream(device=b._device_argument(inputs),
                channels=max(b.config.input_channels), **options)
        b._set_active_stream(self.stream)
        self.latency = self.stream.latency
        self.stream.start()

    def capture(self, frames, output=None):
        from .audio import CaptureResult
        b = self.backend
        requested_at = time.monotonic()
        if output is not None and output.shape[1] > self.outputs:
            raise ValueError('Playback channel count exceeds prepared stream width')
        if self.stream is None:
            b._emit_progress('preparing_device', 0, frames, force=True)
            self.open()
        deadline = time.monotonic() + 15
        while not self.ready and not b._abort_requested.is_set():
            self.check_stream()
            b._emit_progress('preparing_device', 0, frames)
            if time.monotonic() > deadline:
                raise TimeoutError('输入连续性未稳定，尚未开始正式测量；请检查设备/驱动')
            time.sleep(.02)
        status = b._new_callback_status()
        status.update(timing_discontinuity=False, max_callback_gap_seconds=0.,
                      startup_settling_seconds=self.initial_settling_seconds,
                      continuous_experiment_stream=True, stream_latency_seconds=self.latency)
        data = np.zeros((frames, max(b.config.input_channels)), np.float32)
        data.fill(0)
        job = dict(output=output, recording=data, cursor=0, status=status, done=threading.Event(), requested_at=requested_at)
        if not b._abort_requested.is_set():
            self.job = job
            deadline = time.monotonic() + frames / b.config.sample_rate + 15
            try:
                while not job['done'].wait(.02):
                    self.check_stream()
                    b._emit_progress(('playing' if output is not None else 'recording') if job['cursor'] else 'preparing_device', job['cursor'], frames)
                    if b._abort_requested.is_set():
                        # Stop from the owner thread and wait for any in-flight
                        # NumPy copy to finish before reading the partial buffer.
                        self.stream.abort()
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('正式录音超时，输入未持续返回数据')
            finally:
                self.job = None
        cancelled = b._abort_requested.is_set()
        count = job['cursor']
        status.update(processed_frames=count, expected_frames=frames,
                      complete=count == frames and not cancelled, cancelled=cancelled)
        b._emit_progress('cancelled' if cancelled else 'completed', count, frames, force=True)
        result = b._operation_status(b._module(), status)
        result['timing_discontinuity'] = status['timing_discontinuity']
        result['cancelled'] = cancelled
        return CaptureResult(data[:count, np.asarray(b.config.input_channels)-1].copy(), result)

    def check_stream(self):
        if self.error:
            raise RuntimeError(f'音频回调失败：{self.error}') from self.error
        if self.stream is not None and not self.stream.active:
            raise RuntimeError('声卡流意外结束，不能将当前数据标记为有效')

    def close(self):
        self.job = None
        if self.stream is not None:
            try:
                if self.backend._abort_requested.is_set() or self.error:
                    self.stream.abort()
                else:
                    self.stream.stop()
            finally:
                self.stream.close()
                self.backend._set_active_stream(None)
