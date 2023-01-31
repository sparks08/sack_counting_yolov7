import time
from utils.sort import Sort, intersect
import numpy as np
import cv2

DETECTION_FRAME_THICKNESS = 7

OBJECTS_ON_FRAME_COUNTER_FONT = cv2.FONT_HERSHEY_SIMPLEX
OBJECTS_ON_FRAME_COUNTER_FONT_SIZE = 0.5

LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 3
LINE_COUNTER_FONT = cv2.FONT_HERSHEY_DUPLEX
LINE_COUNTER_FONT_SIZE = 2.0
LINE_COUNTER_POSITION = (20, 45)


class CountObjects:

    def __init__(self, line_begin, line_end, names, colors, idle):
        self.tracker = Sort()
        self.memory = {}
        self.counter = 0
        self.names = names
        self.colors = colors
        self.line = [line_begin, line_end]
        self.idle = idle
        self.last_update = time.time() + self.idle

    def _write_quantities(self, frame, labels_quantities_dic):
        for i, (label, quantity) in enumerate(labels_quantities_dic.items()):
            class_id = [i for i, x in enumerate(labels_quantities_dic.keys()) if x == label][0]
            color = [int(c) for c in self.colors[class_id % len(self.colors)]]

            cv2.putText(
                frame,
                f"{label}: {quantity}",
                (10, (i + 1) * 35),
                OBJECTS_ON_FRAME_COUNTER_FONT,
                OBJECTS_ON_FRAME_COUNTER_FONT_SIZE,
                color,
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )

    def _draw_detection_results(self, frame, results, labels_quantities_dic):
        for start_point, end_point, label, confidence in results:
            x1, y1 = start_point

            class_id = [i for i, x in enumerate(labels_quantities_dic.keys()) if x == label][0]

            color = [int(c) for c in self.colors[class_id % len(self.colors)]]

            cv2.rectangle(frame, start_point, end_point, color, DETECTION_FRAME_THICKNESS)

            cv2.putText(frame, label, (x1, y1 - 5), OBJECTS_ON_FRAME_COUNTER_FONT, OBJECTS_ON_FRAME_COUNTER_FONT_SIZE,
                        color, 2)

    def count_objects_in_frame(self, frame, items: dict = {}, targeted_classes: list = []):

        if targeted_classes:
            for k in items.keys():
                if k not in targeted_classes:
                    del items[k]

        return dict([(k, len(v)) for k, v in items.items()])

    def count_objects_crossing_the_virtual_line(self, frame, items: dict = {}, targeted_classes: list = []):

        idle_timeout_reached = False
        count_objects_in_frame = {}
        dets = []
        for (x1, y1, x2, y2, label, confidence) in items:
            if label in targeted_classes:
                dets.append([x1, y1, x2, y2])
                count_objects_in_frame[label] = count_objects_in_frame.get(label, 0) + 1

        # convert to format required for dets [x1, y1, x2, y2, confidence]
        tracks = self.tracker.update(np.asarray(dets))

        boxes = []
        indexIDs = []
        previous = self.memory.copy()
        self.memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            self.memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                color = [int(c) for c in self.colors[indexIDs[i] % len(self.colors)]]

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))

                    if intersect(p0, p1, self.line[0], self.line[1]):
                        frame = cv2.rectangle(frame, (x, y), (w, h), color, DETECTION_FRAME_THICKNESS)
                        #print('---------------------------------------------------------')
                        self.counter += 1
                        print('Object Counting in Progress',self.counter)
                        self.last_update = time.time() + self.idle


        frame = cv2.putText(frame, f"SACKS: {self.counter}", (1500, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)

        if self.last_update < time.time():
            idle_timeout_reached = True
        return frame, idle_timeout_reached

# if __name__ == '__main__':
#     options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights", "threshold": 0.5, "gpu": 1.0}

#     img = cv2.imread("sample_inputs/united_nations.jpg")

#     VIDEO_PATH = "sample_inputs/highway_traffic.mp4"

#     cap = cv2.VideoCapture(VIDEO_PATH)

#     counter = ObjectCountingAPI(options)

#     counter.count_objects_crossing_the_virtual_line(cap, line_begin=(100, 300), line_end=(320, 250), show=True)
#     # counter.count_objects_on_image(img, targeted_classes=["person"], show=True)
