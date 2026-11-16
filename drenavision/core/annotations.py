"""Place short labels without changing detection coordinates."""

from __future__ import annotations


def place_label(
    image_size: tuple[int, int],
    box: tuple[float, float, float, float],
    label_size: tuple[int, int],
    occupied: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    width, height = image_size
    label_width, label_height = label_size
    x1, y1, _x2, y2 = box
    left = max(0, min(x1, width - label_width))
    candidates = [y1 - label_height - 2, y2 + 2, y1 + 2]
    candidates.extend(y1 - label_height * step - 2 for step in range(2, 6))
    for top in candidates:
        top = max(0, min(top, height - label_height))
        rectangle = (left, top, left + label_width, top + label_height)
        overlaps = any(
            rectangle[0] < other[2] + 3
            and rectangle[2] + 3 > other[0]
            and rectangle[1] < other[3] + 3
            and rectangle[3] + 3 > other[1]
            for other in occupied
        )
        if not overlaps:
            occupied.append(rectangle)
            return left, top
    occupied.append((left, y1, left + label_width, y1 + label_height))
    return left, max(0, min(y1, height - label_height))
