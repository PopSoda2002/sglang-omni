from __future__ import annotations

from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class NemotronThinkerScheduler(OmniScheduler):
    @staticmethod
    def _append_stream_chunk_default(req_data, chunk) -> None:
        req_data.acoustic_rows.append(chunk.data)

    def _is_request_build_ready(self, payload, *, pending_stream_done) -> bool:
        del pending_stream_done
        state = NemotronVoiceChatState.from_dict(payload.data)
        if state.acoustic_frames is not None:
            return True
        return len(payload.prefetched_chunks) >= 1

    def _should_recheck_deferred_request_on_stream_chunk(self, request_id, chunk) -> bool:
        del request_id, chunk
        return True

    def get_next_batch_to_run(self):
        if not self.waiting_queue:
            for batch in (self.running_batch, self.last_batch):
                if batch is None or batch.is_empty():
                    continue
                for req in batch.reqs:
                    if req.finished():
                        continue
                    if not self._model_runner.thinker_ready(req._omni_data):
                        return None
        return super().get_next_batch_to_run()

    def self_check_during_idle(self) -> None:
        if self.running_batch is not None and not self.running_batch.is_empty():
            return
        if self.waiting_queue:
            return
        super().self_check_during_idle()
