from __future__ import annotations

import unittest

from src.ms2.inference.beam import beam_decode
from src.ms2.inference.greedy import greedy_decode
from src.ms2.inference.windowing import select_best_window, sliding_context_windows


class TinyDecodable:
    def __init__(self) -> None:
        self.calls = 0

    def init_state(self, encoder_inputs: object) -> dict[str, object]:
        return {"encoder_inputs": encoder_inputs, "steps": 0}

    def step(
        self, state: dict[str, object], last_token: int
    ) -> tuple[list[float], dict[str, object]]:
        del last_token
        steps = int(state["steps"])
        state = {**state, "steps": steps + 1}
        self.calls += 1
        if steps == 0:
            return [-10.0, -10.0, -10.0, -2.0, -0.1], state
        return [-10.0, -10.0, -10.0, -0.1, -2.0], state


class TestInference(unittest.TestCase):
    def test_greedy_stops_at_eos(self) -> None:
        result = greedy_decode(TinyDecodable(), encoder_inputs=[1, 2], max_length=5)
        self.assertEqual(result.token_ids[-1], 3)
        self.assertEqual(len(result.token_ids), 2)

    def test_beam_not_worse_than_greedy(self) -> None:
        model = TinyDecodable()
        greedy = greedy_decode(model, encoder_inputs=[1])
        beam = beam_decode(TinyDecodable(), encoder_inputs=[1], beam_width=4)
        self.assertGreaterEqual(beam.normalized_score, greedy.normalized_score)

    def test_windowing_stride_and_selection(self) -> None:
        windows = sliding_context_windows(list(range(10)), l_c=4)
        self.assertEqual([start for start, _ in windows], [0, 2, 4, 6])
        start, result = select_best_window(
            list(range(10)),
            lambda window: type(
                "R",
                (),
                {
                    "normalized_score": float(window[0]),
                    "token_ids": [],
                    "log_probability": 0.0,
                },
            )(),
            l_c=4,
        )
        self.assertEqual(start, 6)
        self.assertEqual(result.normalized_score, 6.0)


if __name__ == "__main__":
    unittest.main()
