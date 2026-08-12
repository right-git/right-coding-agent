import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screen_locator import Detection, box_center, parse_detections, select_targets


class ParseDetectionsTests(unittest.TestCase):
    def test_reference_label_and_coordinates_are_scaled_to_the_screenshot(self):
        detections = parse_detections(
            "<ref>submit button</ref><box><100><250><900><750></box>",
            (1919, 1079),
        )

        self.assertEqual(
            detections,
            [Detection(label="submit button", box=(192, 270, 1727, 809))],
        )

    def test_multiple_boxes_preserve_order_and_missing_labels_use_object(self):
        detections = parse_detections(
            "prefix <box><0><0><100><100></box>"
            "<ref>icons</ref>"
            "<box><200><300><400><500></box>"
            "<box><600><700><800><900></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [
                Detection(label="object", box=(0, 0, 100, 100)),
                Detection(label="icons", box=(200, 300, 400, 500)),
                Detection(label="icons", box=(600, 700, 800, 900)),
            ],
        )

    def test_malformed_boxes_are_ignored(self):
        detections = parse_detections(
            "<ref>targets</ref>"
            "<box><10><20><30></box>"
            "<box><-10><+20><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="targets", box=(-10, 20, 300, 400))],
        )

    def test_valid_box_after_unterminated_box_is_recovered(self):
        detections = parse_detections(
            "<box><10><20><30>"
            "<box><100><200><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="object", box=(100, 200, 300, 400))],
        )

    def test_labeled_box_after_unterminated_box_is_recovered(self):
        detections = parse_detections(
            "<box><10><20><30>"
            "<ref>good</ref><box><100><200><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="good", box=(100, 200, 300, 400))],
        )

    def test_reference_labels_are_stripped(self):
        detections = parse_detections(
            "<ref>  submit button  </ref><box><100><200><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="submit button", box=(100, 200, 300, 400))],
        )

    def test_empty_reference_label_uses_object(self):
        detections = parse_detections(
            "<ref></ref><box><100><200><300><400></box>",
            (1000, 1000),
        )

        self.assertEqual(
            detections,
            [Detection(label="object", box=(100, 200, 300, 400))],
        )


class BoxCenterTests(unittest.TestCase):
    def test_midpoint_is_returned_as_integer_coordinates(self):
        self.assertEqual(box_center((10, 20, 31, 43), (100, 100)), (20, 31))

    def test_midpoint_is_clamped_to_the_screenshot(self):
        self.assertEqual(box_center((-30, -20, -10, -4), (80, 60)), (0, 0))
        self.assertEqual(box_center((90, 70, 110, 90), (80, 60)), (79, 59))


class SelectTargetsTests(unittest.TestCase):
    def setUp(self):
        self.detections = [
            Detection(label="first", box=(0, 0, 10, 10)),
            Detection(label="second", box=(20, 20, 30, 30)),
        ]

    def test_first_returns_only_the_first_detection(self):
        self.assertEqual(select_targets(self.detections, "first"), self.detections[:1])
        self.assertEqual(select_targets([], "first"), [])

    def test_all_returns_every_detection_in_order(self):
        self.assertEqual(select_targets(self.detections, "all"), self.detections)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported target mode"):
            select_targets(self.detections, "nearest")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
