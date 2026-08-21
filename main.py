# ============================================================
# PRODUCT MOTION V5.2 ANDROID
# Offline Product Video Generator
# Kivy + OpenCV + NumPy + Pillow
# ============================================================

import os
import math
import threading

import cv2
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup

try:
    from PIL import Image
except:
    Image = None


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 540
HEIGHT = 960
FPS = 24


# ============================================================
# EASING
# ============================================================

def ease_in_out(t):

    return (
        0.5 -
        0.5 *
        math.cos(
            math.pi * t
        )
    )


# ============================================================
# RESIZE
# ============================================================

def resize_cover(
        img,
        target_w,
        target_h):

    h, w = img.shape[:2]

    scale = max(
        target_w / w,
        target_h / h
    )

    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    x = max(
        0,
        (nw - target_w) // 2
    )

    y = max(
        0,
        (nh - target_h) // 2
    )

    return resized[
        y:y + target_h,
        x:x + target_w
    ]


# ============================================================
# PRODUCT MASK
# ============================================================

def create_product_mask(image):

    h, w = image.shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    margin_x = int(w * 0.10)
    margin_y = int(h * 0.08)

    rect = (
        margin_x,
        margin_y,
        w - 2 * margin_x,
        h - 2 * margin_y
    )

    bg_model = np.zeros(
        (1, 65),
        np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64
    )

    try:

        cv2.grabCut(
            image,
            mask,
            rect,
            bg_model,
            fg_model,
            3,
            cv2.GC_INIT_WITH_RECT
        )

        result = np.where(
            (mask == cv2.GC_FGD) |
            (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(
            np.uint8
        )

    except Exception:

        result = np.ones(
            (h, w),
            dtype=np.uint8
        ) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    result = cv2.GaussianBlur(
        result,
        (11, 11),
        0
    )

    return result


# ============================================================
# MOTION MASK
# ============================================================

def create_motion_mask(
        product_mask):

    product_binary = (
        product_mask > 80
    ).astype(
        np.uint8
    )

    background = (
        1 -
        product_binary
    )

    distance = cv2.distanceTransform(
        background,
        cv2.DIST_L2,
        5
    )

    maximum = np.max(distance)

    if maximum <= 0:

        return (
            255 -
            product_mask
        ).astype(
            np.float32
        ) / 255.0

    distance /= maximum

    motion = np.clip(
        distance * 1.8,
        0,
        1
    )

    motion[
        product_mask > 80
    ] = 0

    return motion.astype(
        np.float32
    )


# ============================================================
# NATURAL BACKGROUND
# ============================================================

def natural_background_motion(
        image,
        motion_mask,
        t,
        strength):

    h, w = image.shape[:2]

    y_grid, x_grid = np.mgrid[
        0:h,
        0:w
    ].astype(
        np.float32
    )

    wave_x1 = np.sin(
        y_grid * 0.018 +
        t * 1.8
    )

    wave_x2 = np.sin(
        y_grid * 0.009 +
        x_grid * 0.004 +
        t * 1.15
    )

    wave_x3 = np.sin(
        x_grid * 0.012 -
        t * 1.4
    )

    wave_y1 = np.sin(
        x_grid * 0.010 +
        t * 1.4
    )

    displacement_x = (
        wave_x1 * 0.55 +
        wave_x2 * 0.30 +
        wave_x3 * 0.15
    )

    displacement_y = (
        wave_y1 * 0.35
    )

    displacement_x *= (
        strength *
        motion_mask
    )

    displacement_y *= (
        strength *
        motion_mask
    )

    map_x = (
        x_grid +
        displacement_x
    ).astype(
        np.float32
    )

    map_y = (
        y_grid +
        displacement_y
    ).astype(
        np.float32
    )

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


# ============================================================
# CAMERA MOVEMENT
# ============================================================

def camera_transform(
        image,
        progress,
        direction,
        zoom_mode,
        zoom_amount,
        rotation_amount,
        movement_amount):

    h, w = image.shape[:2]

    p = ease_in_out(progress)

    zoom = 1.0

    if zoom_mode == "in":

        zoom = (
            1.0 +
            zoom_amount / 100.0 *
            p
        )

    elif zoom_mode == "out":

        zoom = (
            1.0 +
            zoom_amount / 100.0 *
            (1.0 - p)
        )

    pan_x = 0.0
    pan_y = 0.0

    if direction == "left_right":

        pan_x = (
            -movement_amount +
            2 *
            movement_amount *
            p
        )

    elif direction == "right_left":

        pan_x = (
            movement_amount -
            2 *
            movement_amount *
            p
        )

    elif direction == "up_down":

        pan_y = (
            -movement_amount +
            2 *
            movement_amount *
            p
        )

    elif direction == "down_up":

        pan_y = (
            movement_amount -
            2 *
            movement_amount *
            p
        )

    elif direction == "diagonal_right_up":

        pan_x = (
            -movement_amount +
            2 *
            movement_amount *
            p
        )

        pan_y = (
            movement_amount -
            2 *
            movement_amount *
            p
        )

    elif direction == "diagonal_left_up":

        pan_x = (
            movement_amount -
            2 *
            movement_amount *
            p
        )

        pan_y = (
            movement_amount -
            2 *
            movement_amount *
            p
        )

    drift_x = (
        math.sin(
            p * math.pi * 2
        ) *
        movement_amount *
        0.12
    )

    drift_y = (
        math.sin(
            p * math.pi * 4
        ) *
        movement_amount *
        0.05
    )

    pan_x += drift_x
    pan_y += drift_y

    angle = (
        math.sin(
            p * math.pi * 2
        ) *
        rotation_amount
    )

    matrix = cv2.getRotationMatrix2D(
        (
            w / 2,
            h / 2
        ),
        angle,
        zoom
    )

    matrix[0, 2] += pan_x
    matrix[1, 2] += pan_y

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


# ============================================================
# PRESERVE PRODUCT
# ============================================================

def preserve_product(
        original,
        background,
        product_mask):

    alpha = (
        product_mask.astype(
            np.float32
        ) / 255.0
    )

    alpha = np.expand_dims(
        alpha,
        axis=2
    )

    result = (
        original.astype(
            np.float32
        ) * alpha
        +
        background.astype(
            np.float32
        ) * (1 - alpha)
    )

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# TEXT COMMAND
# ============================================================

def parse_instruction(text):

    text = text.lower().strip()

    result = {
        "direction": None,
        "zoom": None,
        "speed": None,
        "rotation": False
    }

    if (
        "slow" in text or
        "perlahan" in text or
        "cinematic" in text
    ):

        result["speed"] = "slow"

    elif (
        "fast" in text or
        "laju" in text
    ):

        result["speed"] = "fast"

    if (
        "zoom in" in text or
        "zoom-in" in text or
        "dekat" in text
    ):

        result["zoom"] = "in"

    elif (
        "zoom out" in text or
        "zoom-out" in text or
        "jauh" in text
    ):

        result["zoom"] = "out"

    if (
        "left to right" in text or
        "kiri ke kanan" in text
    ):

        result["direction"] = "left_right"

    elif (
        "right to left" in text or
        "kanan ke kiri" in text
    ):

        result["direction"] = "right_left"

    elif (
        "up to down" in text or
        "atas ke bawah" in text
    ):

        result["direction"] = "up_down"

    elif (
        "down to up" in text or
        "bawah ke atas" in text
    ):

        result["direction"] = "down_up"

    elif "diagonal right" in text:

        result["direction"] = (
            "diagonal_right_up"
        )

    elif "diagonal left" in text:

        result["direction"] = (
            "diagonal_left_up"
        )

    if any(
        word in text
        for word in [
            "rotate",
            "rotation",
            "pusing",
            "tilt"
        ]
    ):

        result["rotation"] = True

    return result


# ============================================================
# APP
# ============================================================

class ProductMotionApp(App):

    def build(self):

        self.title = (
            "Product Motion V5.2"
        )

        Window.clearcolor = (
            0.96,
            0.96,
            0.96,
            1
        )

        self.image_path = None

        root = BoxLayout(
            orientation="vertical"
        )

        scroll = ScrollView()

        content = BoxLayout(
            orientation="vertical",
            padding=15,
            spacing=10,
            size_hint_y=None
        )

        content.bind(
            minimum_height=
            content.setter(
                "height"
            )
        )

        title = Label(
            text=(
                "[b]🎬 PRODUCT MOTION V5.2[/b]"
            ),
            markup=True,
            font_size=24,
            size_hint_y=None,
            height=55
        )

        content.add_widget(title)

        subtitle = Label(
            text=(
                "Offline Natural Product Video"
            ),
            size_hint_y=None,
            height=35
        )

        content.add_widget(
            subtitle
        )

        self.preview = KivyImage(
            size_hint_y=None,
            height=300
        )

        content.add_widget(
            self.preview
        )

        select_btn = Button(
            text="📷 SELECT IMAGE",
            size_hint_y=None,
            height=55
        )

        select_btn.bind(
            on_press=self.select_image
        )

        content.add_widget(
            select_btn
        )

        content.add_widget(
            Label(
                text="Duration",
                size_hint_y=None,
                height=30
            )
        )

        self.duration = Spinner(
            text="5 seconds",
            values=(
                "3 seconds",
                "5 seconds",
                "10 seconds"
            ),
            size_hint_y=None,
            height=50
        )

        content.add_widget(
            self.duration
        )

        content.add_widget(
            Label(
                text="Camera Movement",
                size_hint_y=None,
                height=30
            )
        )

        self.direction = Spinner(
            text="Left → Right",
            values=(
                "Auto / Natural",
                "Left → Right",
                "Right → Left",
                "Up → Down",
                "Down → Up",
                "Diagonal Right Up",
                "Diagonal Left Up"
            ),
            size_hint_y=None,
            height=50
        )

        content.add_widget(
            self.direction
        )

        content.add_widget(
            Label(
                text="Zoom",
                size_hint_y=None,
                height=30
            )
        )

        self.zoom = Spinner(
            text="No Zoom",
            values=(
                "No Zoom",
                "Zoom In",
                "Zoom Out"
            ),
            size_hint_y=None,
            height=50
        )

        content.add_widget(
            self.zoom
        )

        content.add_widget(
            Label(
                text="Movement Speed",
                size_hint_y=None,
                height=30
            )
        )

        self.speed = Spinner(
            text="Slow",
            values=(
                "Slow",
                "Normal",
                "Fast"
            ),
            size_hint_y=None,
            height=50
        )

        content.add_widget(
            self.speed
        )

        content.add_widget(
            Label(
                text="Movement Amount",
                size_hint_y=None,
                height=30
            )
        )

        self.movement = Slider(
            min=1,
            max=30,
            value=8,
            size_hint_y=None,
            height=45
        )

        content.add_widget(
            self.movement
        )

        content.add_widget(
            Label(
                text="Zoom Amount (%)",
                size_hint_y=None,
                height=30
            )
        )

        self.zoom_amount = Slider(
            min=0,
            max=15,
            value=5,
            size_hint_y=None,
            height=45
        )

        content.add_widget(
            self.zoom_amount
        )

        content.add_widget(
            Label(
                text="Rotation",
                size_hint_y=None,
                height=30
            )
        )

        self.rotation = Slider(
            min=0,
            max=5,
            value=1,
            size_hint_y=None,
            height=45
        )

        content.add_widget(
            self.rotation
        )

        content.add_widget(
            Label(
                text="Natural Background",
                size_hint_y=None,
                height=30
            )
        )

        self.natural = Slider(
            min=0.5,
            max=5,
            value=1.8,
            size_hint_y=None,
            height=45
        )

        content.add_widget(
            self.natural
        )

        content.add_widget(
            Label(
                text="Movement Instruction",
                size_hint_y=None,
                height=30
            )
        )

        self.instruction = TextInput(
            hint_text=(
                "Contoh: slow zoom in"
            ),
            multiline=True,
            size_hint_y=None,
            height=100
        )

        content.add_widget(
            self.instruction
        )

        self.progress = ProgressBar(
            max=100,
            value=0,
            size_hint_y=None,
            height=25
        )

        content.add_widget(
            self.progress
        )

        self.status = Label(
            text="Ready",
            size_hint_y=None,
            height=40
        )

        content.add_widget(
            self.status
        )

        generate_btn = Button(
            text="🎬 GENERATE VIDEO",
            size_hint_y=None,
            height=65
        )

        generate_btn.bind(
            on_press=self.generate
        )

        content.add_widget(
            generate_btn
        )

        scroll.add_widget(
            content
        )

        root.add_widget(
            scroll
        )

        return root


    # ========================================================
    # SELECT IMAGE
    # ========================================================

    def select_image(
            self,
            instance):

        try:

            from plyer import filechooser

            filechooser.open_file(
                filters=[
                    "*.jpg",
                    "*.jpeg",
                    "*.png",
                    "*.webp"
                ],
                on_selection=
                self.file_selected
            )

        except Exception as e:

            self.status.text = (
                "File picker error: " +
                str(e)
            )


    def file_selected(
            self,
            selection):

        if not selection:
            return

        self.image_path = selection[0]

        try:

            img = cv2.imread(
                self.image_path
            )

            if img is None:
                raise Exception(
                    "Cannot read image"
                )

            img = resize_cover(
                img,
                540,
                540
            )

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            texture = Texture.create(
                size=(
                    img.shape[1],
                    img.shape[0]
                ),
                colorfmt="rgb"
            )

            texture.blit_buffer(
                img.tobytes(),
                colorfmt="rgb",
                bufferfmt="ubyte"
            )

            texture.flip_vertical()

            self.preview.texture = texture

            self.status.text = (
                "Image selected ✓"
            )

        except Exception as e:

            self.status.text = (
                "Image error: " +
                str(e)
            )


    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
            self,
            instance):

        if not self.image_path:

            self.status.text = (
                "Please select image first."
            )

            return

        self.status.text = (
            "Generating..."
        )

        self.progress.value = 0

        thread = threading.Thread(
            target=self.generate_thread
        )

        thread.daemon = True
        thread.start()


    def generate_thread(self):

        try:

            image = cv2.imread(
                self.image_path
            )

            if image is None:

                raise Exception(
                    "Cannot load image"
                )

            image = resize_cover(
                image,
                WIDTH,
                HEIGHT
            )

            # -----------------------------------------------
            # SETTINGS
            # -----------------------------------------------

            duration_text = (
                self.duration.text
            )

            duration = int(
                duration_text.split()[0]
            )

            direction_text = (
                self.direction.text
            )

            direction_map = {

                "Auto / Natural":
                    "left_right",

                "Left → Right":
                    "left_right",

                "Right → Left":
                    "right_left",

                "Up → Down":
                    "up_down",

                "Down → Up":
                    "down_up",

                "Diagonal Right Up":
                    "diagonal_right_up",

                "Diagonal Left Up":
                    "diagonal_left_up"
            }

            direction = direction_map[
                direction_text
            ]

            zoom_map = {

                "No Zoom": "none",
                "Zoom In": "in",
                "Zoom Out": "out"
            }

            zoom_mode = zoom_map[
                self.zoom.text
            ]

            speed_map = {

                "Slow": "slow",
                "Normal": "normal",
                "Fast": "fast"
            }

            speed = speed_map[
                self.speed.text
            ]

            movement_amount = (
                self.movement.value
            )

            zoom_amount = (
                self.zoom_amount.value
            )

            rotation_amount = (
                self.rotation.value
            )

            natural_strength = (
                self.natural.value
            )

            # -----------------------------------------------
            # TEXT COMMAND
            # -----------------------------------------------

            command = parse_instruction(
                self.instruction.text
            )

            if command["direction"]:

                direction = (
                    command["direction"]
                )

            if command["zoom"]:

                zoom_mode = (
                    command["zoom"]
                )

            if command["speed"]:

                speed = (
                    command["speed"]
                )

            if command["rotation"]:

                rotation_amount = max(
                    rotation_amount,
                    1.0
                )

            # -----------------------------------------------
            # PRODUCT MASK
            # -----------------------------------------------

            self.update_status(
                "🔍 Detecting product...",
                5
            )

            product_mask = (
                create_product_mask(
                    image
                )
            )

            motion_mask = (
                create_motion_mask(
                    product_mask
                )
            )

            # -----------------------------------------------
            # SPEED
            # -----------------------------------------------

            speed_factor = 1.0

            if speed == "slow":

                speed_factor = 0.65

            elif speed == "fast":

                speed_factor = 1.45

            # -----------------------------------------------
            # OUTPUT
            # -----------------------------------------------

            output_dir = (
                self.user_data_dir
            )

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            output_file = os.path.join(
                output_dir,
                "ProductMotion_V52.mp4"
            )

            fourcc = (
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                )
            )

            writer = cv2.VideoWriter(
                output_file,
                fourcc,
                FPS,
                (
                    WIDTH,
                    HEIGHT
                )
            )

            if not writer.isOpened():

                raise Exception(
                    "Video encoder unavailable"
                )

            total_frames = int(
                duration * FPS
            )

            # -----------------------------------------------
            # FRAME LOOP
            # -----------------------------------------------

            for frame_number in range(
                total_frames
            ):

                raw_t = (
                    frame_number /
                    max(
                        total_frames - 1,
                        1
                    )
                )

                t = (
                    raw_t *
                    speed_factor
                )

                camera_t = (
                    t % 1.0
                )

                camera_bg = (
                    camera_transform(
                        image,
                        camera_t,
                        direction,
                        zoom_mode,
                        zoom_amount,
                        rotation_amount,
                        movement_amount
                    )
                )

                moving_bg = (
                    natural_background_motion(
                        camera_bg,
                        motion_mask,
                        t,
                        natural_strength
                    )
                )

                frame = (
                    preserve_product(
                        image,
                        moving_bg,
                        product_mask
                    )
                )

                writer.write(
                    frame
                )

                percent = int(
                    10 +
                    85 *
                    (
                        frame_number + 1
                    ) /
                    total_frames
                )

                self.update_status(
                    "🎞️ Generating " +
                    str(frame_number + 1) +
                    "/" +
                    str(total_frames),
                    percent
                )

            writer.release()

            # -----------------------------------------------
            # COPY TO GALLERY
            # -----------------------------------------------

            self.update_status(
                "💾 Saving video...",
                98
            )

            gallery_path = (
                self.save_to_gallery(
                    output_file
                )
            )

            self.update_status(
                "✅ Video saved!",
                100
            )

            Clock.schedule_once(
                lambda dt:
                self.show_done(
                    gallery_path
                ),
                0
            )

        except Exception as e:

            Clock.schedule_once(
                lambda dt:
                self.show_error(
                    str(e)
                ),
                0
            )


    # ========================================================
    # STATUS
    # ========================================================

    def update_status(
            self,
            text,
            progress):

        Clock.schedule_once(
            lambda dt:
            self.set_status(
                text,
                progress
            ),
            0
        )


    def set_status(
            self,
            text,
            progress):

        self.status.text = text
        self.progress.value = progress


    # ========================================================
    # SAVE GALLERY
    # ========================================================

    def save_to_gallery(
            self,
            source):

        try:

            from jnius import autoclass

            Environment = autoclass(
                "android.os.Environment"
            )

            MediaStore = autoclass(
                "android.provider.MediaStore"
            )

            PythonActivity = autoclass(
                "org.kivy.android.PythonActivity"
            )

            activity = (
                PythonActivity.mActivity
            )

            resolver = (
                activity.getContentResolver()
            )

            ContentValues = autoclass(
                "android.content.ContentValues"
            )

            values = ContentValues()

            values.put(
                MediaStore.MediaColumns.DISPLAY_NAME,
                "ProductMotion_V52.mp4"
            )

            values.put(
                MediaStore.MediaColumns.MIME_TYPE,
                "video/mp4"
            )

            values.put(
                MediaStore.MediaColumns.RELATIVE_PATH,
                "Movies/ProductMotion"
            )

            collection = (
                MediaStore.Video.Media
                .getContentUri(
                    "external"
                )
            )

            uri = resolver.insert(
                collection,
                values
            )

            if uri is None:

                return source

            output = resolver.openOutputStream(
                uri
            )

            with open(
                source,
                "rb"
            ) as f:

                while True:

                    data = f.read(
                        1024 * 1024
                    )

                    if not data:
                        break

                    output.write(
                        data
                    )

            output.close()

            return (
                "Movies/ProductMotion/" +
                "ProductMotion_V52.mp4"
            )

        except Exception:

            return source


    # ========================================================
    # POPUP
    # ========================================================

    def show_done(
            self,
            path):

        Popup(
            title="🎉 Success",
            content=Label(
                text=(
                    "Video berjaya dibuat!\n\n"
                    "Disimpan dalam:\n"
                    "Movies/ProductMotion"
                )
            ),
            size_hint=(
                0.85,
                0.35
            )
        ).open()


    def show_error(
            self,
            error):

        self.status.text = (
            "❌ Error"
        )

        Popup(
            title="Video Generation Error",
            content=Label(
                text=str(error)
            ),
            size_hint=(
                0.9,
                0.5
            )
        ).open()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    ProductMotionApp().run()
