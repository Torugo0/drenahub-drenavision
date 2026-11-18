from __future__ import annotations

import numpy as np

from drenavision.core.schemas import Detection
from drenavision.inference import video


def test_later_box_edge_cannot_cross_an_earlier_label(monkeypatch):
    image = np.full((210, 320, 3), 220, np.uint8)
    first = Detection(
        class_id=0,
        class_name="obstrucao_visivel",
        confidence=0.9,
        bounding_box=(20, 80, 180, 150),
    )
    # Its left edge crosses the position of the first detection's text label.
    crossing = Detection(
        class_id=2,
        class_name="animal_visivel",
        confidence=0.85,
        bounding_box=(55, 20, 240, 170),
    )
    label_rectangles = []
    original = video.place_label

    def record_label(image_size, box, label_size, occupied):
        left, top = original(image_size, box, label_size, occupied)
        label_rectangles.append(
            (int(left), int(top), int(left + label_size[0]), int(top + label_size[1]))
        )
        return left, top

    monkeypatch.setattr(video, "place_label", record_label)
    single = video.annotate_video_frame(image, [first])
    x1, y1, x2, y2 = label_rectangles[0]
    before = [first.model_dump(), crossing.model_dump()]
    multiple = video.annotate_video_frame(image, [first, crossing])
    assert x1 < crossing.bounding_box[0] < x2
    assert crossing.bounding_box[1] < y1 < y2 < crossing.bounding_box[3]
    # The entire filled text label remains visually identical, despite the
    # later purple rectangle passing through it. The old draw order failed here.
    assert np.array_equal(
        single[y1 : y2 + 1, x1 : x2 + 1], multiple[y1 : y2 + 1, x1 : x2 + 1]
    )
    assert [first.model_dump(), crossing.model_dump()] == before
    assert np.array_equal(image, np.full_like(image, 220))
