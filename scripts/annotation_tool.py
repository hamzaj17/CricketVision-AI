import cv2
import os
import copy

IMAGE_FOLDER = "datasets/remaining_dataset/images"
LABEL_FOLDER = "datasets/remaining_dataset/labels"

CLASSES = [
    "ball",
    "bat",
    "player",
    "stumps"
]

COLORS = {
    0: (0, 255, 255),    # Ball
    1: (255, 0, 0),      # Bat
    2: (0, 255, 0),      # Player
    3: (0, 0, 255)       # Stumps
}

WINDOW_NAME = "CricketVision Annotation Tool"

class Annotation:

    def __init__(self, class_id, x1, y1, x2, y2):

        self.class_id = class_id

        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    def contains(self, x, y):

        return (
            self.x1 <= x <= self.x2 and
            self.y1 <= y <= self.y2
        )

    def to_yolo(self, width, height):

        xc = ((self.x1 + self.x2) / 2) / width
        yc = ((self.y1 + self.y2) / 2) / height

        w = abs(self.x2 - self.x1) / width
        h = abs(self.y2 - self.y1) / height

        return (
            self.class_id,
            xc,
            yc,
            w,
            h
        )

class Dataset:

    def __init__(self):

        self.images = sorted([

            f for f in os.listdir(IMAGE_FOLDER)

            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )

        ])

        self.index = 0

    def current_image_name(self):

        return self.images[self.index]

    def current_image_path(self):

        return os.path.join(
            IMAGE_FOLDER,
            self.current_image_name()
        )

    def current_label_path(self):

        name = os.path.splitext(
            self.current_image_name()
        )[0]

        return os.path.join(
            LABEL_FOLDER,
            name + ".txt"
        )

    def next(self):

        if self.index < len(self.images) - 1:
            self.index += 1

    def previous(self):

        if self.index > 0:
            self.index -= 1

class AnnotationTool:

    def __init__(self):

        self.dataset = Dataset()

        self.image = None
        self.display = None

        self.width = 0
        self.height = 0

        self.annotations = []

        self.selected = -1

        self.dragging = False
        self.drawing = False

        self.start_x = 0
        self.start_y = 0

        self.current_class = 2

        self.undo_stack = []

        cv2.namedWindow(WINDOW_NAME)

        cv2.setMouseCallback(
            WINDOW_NAME,
            self.mouse_event
        )

        self.load_current_image()

    def load_current_image(self):

        path = self.dataset.current_image_path()

        self.image = cv2.imread(path)

        self.height, self.width = self.image.shape[:2]

        self.load_labels()


    def load_labels(self):

        self.annotations.clear()

        label_path = self.dataset.current_label_path()

        if not os.path.exists(label_path):
            return

        with open(label_path, "r") as f:

            for line in f.readlines():

                values = line.strip().split()

                if len(values) != 5:
                    continue

                cls = int(values[0])

                xc = float(values[1])
                yc = float(values[2])

                w = float(values[3])
                h = float(values[4])

                x1 = int((xc - w / 2) * self.width)
                y1 = int((yc - h / 2) * self.height)

                x2 = int((xc + w / 2) * self.width)
                y2 = int((yc + h / 2) * self.height)

                self.annotations.append(

                    Annotation(
                        cls,
                        x1,
                        y1,
                        x2,
                        y2
                    )

                )


    def save_labels(self):

        label_path = self.dataset.current_label_path()

        with open(label_path, "w") as f:

            for ann in self.annotations:

                cls, xc, yc, w, h = ann.to_yolo(
                    self.width,
                    self.height
                )

                f.write(

                    f"{cls} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}\n"

                )

        print(
            f"Saved -> {os.path.basename(label_path)}"
        )


    def backup(self):

        self.undo_stack.append(
            copy.deepcopy(self.annotations)
        )

        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)


    def undo(self):

        if len(self.undo_stack) == 0:
            return

        self.annotations = self.undo_stack.pop()

        self.selected = -1


    def select_box(self, x, y):

        self.selected = -1

        # Search from top-most annotation
        for i in range(len(self.annotations) - 1, -1, -1):

            if self.annotations[i].contains(x, y):

                self.selected = i
                return

    def delete_selected(self):

        if self.selected == -1:
            return

        self.backup()

        self.annotations.pop(self.selected)

        self.selected = -1

    def mouse_event(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.select_box(x, y)

            # Existing box selected
            if self.selected != -1:

                self.dragging = True

                self.start_x = x
                self.start_y = y

                return

            self.drawing = True

            self.start_x = x
            self.start_y = y

        # MOUSE MOVE
        elif event == cv2.EVENT_MOUSEMOVE:

            # Move selected box

            if self.dragging and self.selected != -1:

                dx = x - self.start_x
                dy = y - self.start_y

                ann = self.annotations[self.selected]

                ann.x1 += dx
                ann.x2 += dx

                ann.y1 += dy
                ann.y2 += dy

                self.start_x = x
                self.start_y = y

        # LEFT BUTTON UP

        elif event == cv2.EVENT_LBUTTONUP:

            # Finish dragging

            if self.dragging:

                self.dragging = False

                return

            if self.drawing:

                self.drawing = False

                x1 = min(self.start_x, x)
                y1 = min(self.start_y, y)

                x2 = max(self.start_x, x)
                y2 = max(self.start_y, y)

                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:

                    self.backup()

                    self.annotations.append(

                        Annotation(
                            self.current_class,
                            x1,
                            y1,
                            x2,
                            y2
                        )

                    )

        # RIGHT CLICK

        elif event == cv2.EVENT_RBUTTONDOWN:

            self.selected = -1

            self.dragging = False

            self.drawing = False

    # DRAW

    def draw(self):

        self.display = self.image.copy()

        # Draw all annotations

        for i, ann in enumerate(self.annotations):

            color = COLORS[ann.class_id]

            if i == self.selected:

                color = (0, 255, 255)

            cv2.rectangle(

                self.display,

                (ann.x1, ann.y1),

                (ann.x2, ann.y2),

                color,

                2

            )

            cv2.putText(

                self.display,

                CLASSES[ann.class_id],

                (ann.x1, max(20, ann.y1 - 6)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                color,

                2

            )

        cv2.rectangle(

            self.display,

            (0, 0),

            (self.width, 45),

            (40, 40, 40),

            -1

        )

        text = (

            f"{self.dataset.index+1}/{len(self.dataset.images)}   "

            f"{self.dataset.current_image_name()}"

        )

        cv2.putText(

            self.display,

            text,

            (10, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255,255,255),

            2

        )

        cv2.putText(

            self.display,

            f"Current: {CLASSES[self.current_class]}",

            (650,30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            COLORS[self.current_class],

            2

        )

        cv2.imshow(

            WINDOW_NAME,

            self.display

        )

    # CHANGE CURRENT CLASS

    def set_class(self, class_id):

        if 0 <= class_id < len(CLASSES):
            self.current_class = class_id

            if self.selected != -1:
                self.backup()
                self.annotations[self.selected].class_id = class_id

    # NEXT IMAGE

    def next_image(self):

        self.save_labels()

        self.selected = -1

        self.dataset.next()

        self.load_current_image()


    # PREVIOUS IMAGE
    def previous_image(self):

        self.save_labels()

        self.selected = -1

        self.dataset.previous()

        self.load_current_image()

    def run(self):

        while True:

            self.draw()

            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if key == 27:

                self.save_labels()

                break

            elif key == ord('s'):

                self.save_labels()


            elif key == ord('n'):

                self.next_image()


            elif key == ord('p'):

                self.previous_image()


            elif key in [8, 127]:

                self.delete_selected()


            elif key == ord('u'):

                self.undo()

            elif key == ord('1'):

                self.set_class(0)

            elif key == ord('2'):

                self.set_class(1)

            elif key == ord('3'):

                self.set_class(2)

            elif key == ord('4'):

                self.set_class(3)

        cv2.destroyAllWindows()

if __name__ == "__main__":

    tool = AnnotationTool()

    tool.run()