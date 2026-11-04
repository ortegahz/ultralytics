"""历史环形缓冲的契约测试。"""

from __future__ import annotations

import unittest

import numpy as np

from manu.tracking.async_dual_rate.history import HistoryRingBuffer


class HistoryRingBufferTest(unittest.TestCase):
    def test_push_and_get(self):
        buffer = HistoryRingBuffer(capacity=8)
        buffer.push(0, (10.0, 20.0))
        buffer.push(1, (11.0, 21.0))
        self.assertEqual(len(buffer), 2)
        np.testing.assert_allclose(buffer.get(1), [11.0, 21.0])
        self.assertIsNone(buffer.get(99))

    def test_overwrites_oldest_when_full(self):
        """定容环形缓冲：满了覆盖最老的，旧的取不到了。"""
        buffer = HistoryRingBuffer(capacity=4)
        for index in range(10):
            buffer.push(index, (float(index), 0.0))
        self.assertEqual(len(buffer), 4)
        self.assertIsNone(buffer.get(5))
        self.assertIsNotNone(buffer.get(9))
        np.testing.assert_allclose(buffer.frames(), [6, 7, 8, 9])

    def test_velocity_is_per_frame_difference(self):
        buffer = HistoryRingBuffer(capacity=8)
        buffer.push(0, (0.0, 0.0))
        buffer.push(1, (2.0, -1.0))
        buffer.push(2, (4.0, -2.0))
        np.testing.assert_allclose(buffer.get_velocity(1), [2.0, -1.0])
        np.testing.assert_allclose(buffer.get_velocity(2), [2.0, -1.0])

    def test_velocity_zero_on_frame_gap(self):
        """断帧 ⇒ 速度未知（置零），外推就是"保持不动"，不能瞎猜。"""
        buffer = HistoryRingBuffer(capacity=8)
        buffer.push(0, (0.0, 0.0))
        buffer.push(5, (10.0, 0.0))
        np.testing.assert_allclose(buffer.get_velocity(5), [0.0, 0.0])

    def test_earliest_and_latest(self):
        buffer = HistoryRingBuffer(capacity=4)
        for index in range(3):
            buffer.push(index, (float(index), 0.0))
        self.assertEqual(buffer.oldest(), 0)
        self.assertEqual(buffer.newest(), 2)


if __name__ == "__main__":
    unittest.main()
