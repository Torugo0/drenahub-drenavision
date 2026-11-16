from io import BytesIO

from PIL import Image

from drenavision.ui.streamlit_app import annotate, result_counts


def test_phone_photo_boxes_use_the_same_exif_orientation_as_inference():
    image = Image.new("RGB", (160, 90), "white")
    exif = Image.Exif()
    exif[274] = 6  # Portrait photograph stored sideways by a phone.
    encoded = BytesIO()
    image.save(encoded, format="JPEG", exif=exif)
    marked = annotate(
        encoded.getvalue(),
        [{"class_name": "animal_visivel", "bounding_box": [10, 40, 70, 120]}],
    )
    assert marked.size == (90, 160)
    assert marked.getpixel((10, 100)) == (124, 58, 237)


def test_video_counts_do_not_sum_repeated_frame_detections():
    payload = {
        "counts": {"animal_visivel": {"count": 1}},
        "frames": [{"detections": [{"class_name": "animal_visivel"}]}] * 50,
    }
    assert result_counts(payload, "video") == {
        "obstrucao_visivel": 0,
        "agua_acumulada": 0,
        "animal_visivel": 1,
    }


def test_image_counts_include_all_detections_and_missing_classes():
    payload = {
        "detections": [
            {"class_name": "obstrucao_visivel"},
            {"class_name": "obstrucao_visivel"},
            {"class_name": "animal_visivel"},
        ]
    }
    assert result_counts(payload, "image") == {
        "obstrucao_visivel": 2,
        "agua_acumulada": 0,
        "animal_visivel": 1,
    }
