import tkinter

from ultralytics import YOLO
import cv2
import os
import datetime
import csv
import pandas as pd
from tracker import*
from tkinter import Tk, Frame, Canvas, PhotoImage, Button, Entry, filedialog, messagebox, Listbox, Scrollbar, ttk
from pathlib import Path
from pygrabber.dshow_graph import FilterGraph


# TRAIN, VALIDATION, AND TESTING OF THE DATA AND MODEL
def create_model():
    model = YOLO("yolov8n.pt")  # build a new model from scratch
    model.train(data="data.yaml", epochs=100, batch=16, imgsz=640, project='./model_results', name='train3.0')
    model.val(data='data.yaml', project='./model_results', name='val4.0')
    model.predict('./test3.0/images', save=True, save_txt=True, conf=0.7, project='./model_results', name='test3.0')


# FOR USER INTERFACE
class count_vehicles:
    def __init__(self, master):
        self.master = master
        self.master.geometry("700x550")
        self.master.configure(bg="#000E1D")

        self.home_page_frame = Frame(master, bg="#000E1D")
        self.home_page_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.select_video_frame = Frame(master, bg="#000E1D")
        self.select_video_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.select_camera_frame = Frame(master, bg="#000E1D")
        self.select_camera_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.modifications_coord_frame = Frame(master, bg="#000E1D")
        self.modifications_coord_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.model_predict_frame = Frame(master, bg="#000E1D")
        self.model_predict_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.print_csv_in_content_frame = Frame(master, bg="#000E1D")
        self.print_csv_in_content_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.print_csv_out_content_frame = Frame(master, bg="#000E1D")
        self.print_csv_out_content_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.print_csv_all_content_frame = Frame(master, bg="#000E1D")
        self.print_csv_all_content_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.exit_page_frame = Frame(master, bg="#000E1D")
        self.exit_page_frame.place(x=0, y=0, relwidth=1, relheight=1)

        all_header = ['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                      'IN_COUNT', 'OUT_COUNT', 'TOTAL', 'END_TIME']

        in_header = ['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                     'IN_TOTAL', 'END_TIME']

        out_header = ['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                      'OUT_TOTAL', 'END_TIME']

        self.create_csv_file(csv_file_path, all_header)
        self.create_csv_file(in_csv_file_path, in_header)
        self.create_csv_file(out_csv_file_path, out_header)

        self.home_page()

        master.resizable(False, False)

    def create_csv_file(self, file_path, header):
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)

    def home_page(self):
        self.home_page_frame.destroy()
        self.home_page_frame = Frame(self.master, bg="#000E1D")
        self.home_page_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = Canvas(
            self.home_page_frame,
            bg="#000E1D",
            height=550,
            width=700,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets0("image_1.png"))
        self.image_1 = self.canvas.create_image(
            350.0,
            275.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets0("image_2.png"))
        self.image_2 = self.canvas.create_image(
            503.0,
            275.0,
            image=self.image_image_2
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets0("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.select_video,
            relief="flat"
        )
        self.button_1.place(
            x=292.0,
            y=300.0,
            width=408.0,
            height=56.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets0("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=self.select_camera,
            relief="flat"
        )
        self.button_2.place(
            x=292.0,
            y=368.0,
            width=408.0,
            height=56.0
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets0("image_3.png"))
        self.image_3 = self.canvas.create_image(
            420.0,
            259.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets0("image_4.png"))
        self.image_4 = self.canvas.create_image(
            402.0,
            168.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(
            file=self.relative_to_assets0("image_5.png"))
        self.image_5 = self.canvas.create_image(
            495.0,
            96.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=self.relative_to_assets0("image_6.png"))
        self.image_6 = self.canvas.create_image(
            428.0,
            76.0,
            image=self.image_image_6
        )

        self.image_image_7 = PhotoImage(
            file=self.relative_to_assets0("image_7.png"))
        self.image_7 = self.canvas.create_image(
            348.0,
            55.0,
            image=self.image_image_7
        )

        self.image_image_8 = PhotoImage(
            file=self.relative_to_assets0("image_8.png"))
        self.image_8 = self.canvas.create_image(
            629.0,
            52.0,
            image=self.image_image_8
        )

    def relative_to_assets0(self, path: str) -> Path:
        return home_asset_path / Path(path)

    # MODE SELECTION
    def select_video(self):
        self.select_video_frame.destroy()
        self.select_video_frame = Frame(self.master, bg="#000E1D")
        self.select_video_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = Canvas(
            self.select_video_frame,
            bg="#000E1D",
            height=550,
            width=700,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets1("image_1.png"))
        self.image_1 = self.canvas.create_image(
            350.0,
            275.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets1("image_2.png"))
        self.image_2 = self.canvas.create_image(
            503.0,
            275.0,
            image=self.image_image_2
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets1("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.modifications_coord(self.source, 1),
            relief="flat"
        )
        self.button_1.place(
            x=292.0,
            y=377.0,
            width=408.0,
            height=56.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets1("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.home_page(),
            relief="flat"
        )
        self.button_2.place(
            x=292.0,
            y=444.0,
            width=408.0,
            height=56.0
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets1("image_3.png"))
        self.image_3 = self.canvas.create_image(
            422.0,
            179.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets1("image_4.png"))
        self.image_4 = self.canvas.create_image(
            495.0,
            96.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(
            file=self.relative_to_assets1("image_5.png"))
        self.image_5 = self.canvas.create_image(
            428.0,
            76.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=self.relative_to_assets1("image_6.png"))
        self.image_6 = self.canvas.create_image(
            348.0,
            55.0,
            image=self.image_image_6
        )

        self.image_image_7 = PhotoImage(
            file=self.relative_to_assets1("image_7.png"))
        self.image_7 = self.canvas.create_image(
            643.0,
            52.0,
            image=self.image_image_7
        )

        self.entry_image_1 = PhotoImage(
            file=self.relative_to_assets1("entry_1.png"))
        self.entry_bg_1 = self.canvas.create_image(
            489.5,
            244.5,
            image=self.entry_image_1
        )
        self.entry_1 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0
        )
        self.entry_1.place(
            x=331.0,
            y=218.0,
            width=317.0,
            height=51.0
        )

        self.source = "./vid_test"
        def locate_video():
            self.source = tkinter.filedialog.askopenfilename()
            self.entry_1.delete(0, tkinter.END)
            self.entry_1.insert(0, self.source)

        self.button_image_3 = PhotoImage(
            file=self.relative_to_assets1("button_3.png"))
        self.button_3 = Button(
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: locate_video(),
            relief="flat"
        )
        self.button_3.place(
            x=610.0,
            y=223.0,
            width=46.0,
            height=44.0
        )

    def relative_to_assets1(self, path: str) -> Path:
        return select_video_asset_path / Path(path)

    # SELECT CAMERA DEVICES
    def select_camera(self):
        self.select_camera_frame.destroy()
        self.select_camera_frame = Frame(self.master, bg="#000E1D")
        self.select_camera_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = Canvas(
            self.select_camera_frame,
            bg="#000E1D",
            height=550,
            width=700,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets2("image_1.png"))
        self.image_1 = self.canvas.create_image(
            350.0,
            275.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets2("image_2.png"))
        self.image_2 = self.canvas.create_image(
            503.0,
            275.0,
            image=self.image_image_2
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets2("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.get_camera_input(),
            relief="flat"
        )
        self.button_1.place(
            x=292.0,
            y=377.0,
            width=408.0,
            height=56.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets2("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.home_page(),
            relief="flat"
        )
        self.button_2.place(
            x=292.0,
            y=444.0,
            width=408.0,
            height=56.0
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets2("image_3.png"))
        self.image_3 = self.canvas.create_image(
            441.0,
            179.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets2("image_4.png"))
        self.image_4 = self.canvas.create_image(
            495.0,
            96.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(
            file=self.relative_to_assets2("image_5.png"))
        self.image_5 = self.canvas.create_image(
            428.0,
            76.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=self.relative_to_assets2("image_6.png"))
        self.image_6 = self.canvas.create_image(
            348.0,
            55.0,
            image=self.image_image_6
        )

        self.image_image_7 = PhotoImage(
            file=self.relative_to_assets2("image_7.png"))
        self.image_7 = self.canvas.create_image(
            641.0,
            52.0,
            image=self.image_image_7
        )

        self.device_listbox = Listbox(self.select_camera_frame, selectmode="single", justify="center")
        self.device_listbox.place(x=313.0, y=200.0, width=360.0, height=130.0)
        self.scrollbar = Scrollbar(self.select_camera_frame, command=self.device_listbox.yview)
        self.scrollbar.place(x=660.0, y=200.0, height=130.0)
        self.device_listbox.config(yscrollcommand=self.scrollbar.set)

        self.refresh_button = Button(self.select_camera_frame, text="Refresh", command=self.refresh_devices)
        self.refresh_button.place(x=500.0, y=340.0, width=170.0, height=30)

        self.refresh_devices()

    def refresh_devices(self):
        self.device_listbox.delete(0, "end")
        cameras = self.get_camera_devices()
        for idx, info in enumerate(cameras, start=1):
            self.device_listbox.insert("end", f"{info}")

    def get_camera_devices(self):
        try:
            # Use pygrabber to get the camera information
            camera_info = []
            devices = FilterGraph().get_input_devices()

            for device_index, device_name in enumerate(devices):
                camera_info.append(f"{device_index}: {device_name}")

            return camera_info
        except Exception as e:
            print(f"Error: {e}")
            return []

    def get_camera_input(self):
        selected_index = self.device_listbox.curselection()
        if selected_index:
            devices = self.get_camera_devices()
            if 0 <= int(selected_index[0]) <= len(devices):
                self.modifications_coord(selected_index[0], 2)
            else:
                messagebox.showwarning("Invalid Index", "Please enter a valid camera device index.")
        else:
            messagebox.showwarning("No Camera Selected", "Please select a camera.")

    def relative_to_assets2(self, path: str) -> Path:
        return select_camera_asset_path / Path(path)

    def modifications_coord(self, source, previous):
        self.modifications_coord_frame.destroy()
        self.modifications_coord_frame = Frame(self.master, bg="#000E1D")
        self.modifications_coord_frame.place(x=0, y=0, relwidth=1, relheight=1)

        # if len(source) > 2:
        #     msource = str(source)
        # else:
        #     msource = int(source)

        self.canvas = Canvas(
            self.modifications_coord_frame,
            bg="#FFFFFF",
            height=550,
            width=700,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets3("image_1.png"))
        self.image_1 = self.canvas.create_image(
            253.0,
            275.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets3("image_2.png"))
        self.image_2 = self.canvas.create_image(
            503.0,
            275.0,
            image=self.image_image_2
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets3("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.check_frame(source),
            relief="flat"
        )
        self.button_1.place(
            x=292.0,
            y=377.0,
            width=408.0,
            height=45.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets3("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.model_predict(source),
            relief="flat"
        )
        self.button_2.place(
            x=292.0,
            y=427.0,
            width=408.0,
            height=45.0
        )

        self.button_image_3 = PhotoImage(
            file=self.relative_to_assets3("button_3.png"))
        self.button_3 = Button(
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.select_video() if previous == 1 else self.select_camera(),
            relief="flat"
        )
        self.button_3.place(
            x=292.0,
            y=477.0,
            width=408.0,
            height=45.0
        )

        self.entry_image_1 = PhotoImage(
            file=self.relative_to_assets3("entry_1.png"))
        self.entry_bg_1 = self.canvas.create_image(
            428.0,
            226.5,
            image=self.entry_image_1
        )
        self.entry_1 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_1.insert(0, "1020")
        self.entry_1.place(
            x=389.0,
            y=206.0,
            width=78.0,
            height=39.0
        )

        self.entry_image_2 = PhotoImage(
            file=self.relative_to_assets3("entry_2.png"))
        self.entry_bg_2 = self.canvas.create_image(
            353.0,
            308.0,
            image=self.entry_image_2
        )
        self.entry_2 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_2.insert(0, "0")
        self.entry_2.place(
            x=339.0,
            y=292.0,
            width=28.0,
            height=30.0
        )

        self.entry_image_3 = PhotoImage(
            file=self.relative_to_assets3("entry_3.png"))
        self.entry_bg_3 = self.canvas.create_image(
            549.0,
            308.0,
            image=self.entry_image_3
        )
        self.entry_3 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_3.insert(0, "0")
        self.entry_3.place(
            x=535.0,
            y=292.0,
            width=28.0,
            height=30.0
        )

        self.entry_image_4 = PhotoImage(
            file=self.relative_to_assets3("entry_4.png"))
        self.entry_bg_4 = self.canvas.create_image(
            443.0,
            308.0,
            image=self.entry_image_4
        )
        self.entry_4 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_4.insert(0, "235")
        self.entry_4.place(
            x=429.0,
            y=292.0,
            width=28.0,
            height=30.0
        )

        self.entry_image_5 = PhotoImage(
            file=self.relative_to_assets3("entry_5.png"))
        self.entry_bg_5 = self.canvas.create_image(
            639.0,
            308.0,
            image=self.entry_image_5
        )
        self.entry_5 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_5.insert(0, "300")
        self.entry_5.place(
            x=625.0,
            y=292.0,
            width=28.0,
            height=30.0
        )

        self.entry_image_6 = PhotoImage(
            file=self.relative_to_assets3("entry_6.png"))
        self.entry_bg_6 = self.canvas.create_image(
            353.0,
            342.0,
            image=self.entry_image_6
        )
        self.entry_6 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_6.insert(0, "1020")
        self.entry_6.place(
            x=339.0,
            y=326.0,
            width=28.0,
            height=30.0
        )

        self.entry_image_7 = PhotoImage(
            file=self.relative_to_assets3("entry_7.png"))
        self.entry_bg_7 = self.canvas.create_image(
            549.0,
            342.0,
            image=self.entry_image_7
        )
        self.entry_7 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_7.insert(0, "1020")
        self.entry_7.place(
            x=535.0,
            y=326.0,
            width=28.0,
            height=30.0
        )

        self.entry_image_8 = PhotoImage(
            file=self.relative_to_assets3("entry_8.png"))
        self.entry_bg_8 = self.canvas.create_image(
            443.0,
            342.0,
            image=self.entry_image_8
        )
        self.entry_8 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_8.insert(0, "235")
        self.entry_8.place(
            x=429.0,
            y=326.0,
            width=28.0,
            height=30.0
        )

        self.entry_image_9 = PhotoImage(
            file=self.relative_to_assets3("entry_9.png"))
        self.entry_bg_9 = self.canvas.create_image(
            639.0,
            342.0,
            image=self.entry_image_9
        )
        self.entry_9 = Entry(
            bd=0,
            bg="#43DBD2",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_9.insert(0, "300")
        self.entry_9.place(
            x=625.0,
            y=326.0,
            width=28.0,
            height=30.0
        )

        self.entry_image_10 = PhotoImage(
            file=self.relative_to_assets3("entry_10.png"))
        self.entry_bg_10 = self.canvas.create_image(
            615.5,
            226.5,
            image=self.entry_image_10
        )
        self.entry_10 = Entry(
            bd=0,
            bg="#42DAD1",
            fg="#000716",
            highlightthickness=0,
            validate="key",
            validatecommand=(self.master.register(self.validate_input), "%P")
        )
        self.entry_10.insert(0, "500")
        self.entry_10.place(
            x=579.0,
            y=206.0,
            width=73.0,
            height=39.0
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets3("image_3.png"))
        self.image_3 = self.canvas.create_image(
            440.0,
            130.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets3("image_4.png"))
        self.image_4 = self.canvas.create_image(
            340.0,
            189.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(
            file=self.relative_to_assets3("image_5.png"))
        self.image_5 = self.canvas.create_image(
            473.0,
            155.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=self.relative_to_assets3("image_6.png"))
        self.image_6 = self.canvas.create_image(
            342.0,
            279.0,
            image=self.image_image_6
        )

        self.image_image_7 = PhotoImage(
            file=self.relative_to_assets3("image_7.png"))
        self.image_7 = self.canvas.create_image(
            532.0,
            279.0,
            image=self.image_image_7
        )

        self.image_image_8 = PhotoImage(
            file=self.relative_to_assets3("image_8.png"))
        self.image_8 = self.canvas.create_image(
            335.0,
            226.0,
            image=self.image_image_8
        )

        self.image_image_9 = PhotoImage(
            file=self.relative_to_assets3("image_9.png"))
        self.image_9 = self.canvas.create_image(
            311.0,
            308.0,
            image=self.image_image_9
        )

        self.image_image_10 = PhotoImage(
            file=self.relative_to_assets3("image_10.png"))
        self.image_10 = self.canvas.create_image(
            507.0,
            308.0,
            image=self.image_image_10
        )

        self.image_image_11 = PhotoImage(
            file=self.relative_to_assets3("image_11.png"))
        self.image_11 = self.canvas.create_image(
            312.0,
            342.0,
            image=self.image_image_11
        )

        self.image_image_12 = PhotoImage(
            file=self.relative_to_assets3("image_12.png"))
        self.image_12 = self.canvas.create_image(
            508.0,
            342.0,
            image=self.image_image_12
        )

        self.image_image_13 = PhotoImage(
            file=self.relative_to_assets3("image_13.png"))
        self.image_13 = self.canvas.create_image(
            398.0,
            308.0,
            image=self.image_image_13
        )

        self.image_image_14 = PhotoImage(
            file=self.relative_to_assets3("image_14.png"))
        self.image_14 = self.canvas.create_image(
            594.0,
            308.0,
            image=self.image_image_14
        )

        self.image_image_15 = PhotoImage(
            file=self.relative_to_assets3("image_15.png"))
        self.image_15 = self.canvas.create_image(
            399.0,
            345.0,
            image=self.image_image_15
        )

        self.image_image_16 = PhotoImage(
            file=self.relative_to_assets3("image_16.png"))
        self.image_16 = self.canvas.create_image(
            595.0,
            345.0,
            image=self.image_image_16
        )

        self.image_image_17 = PhotoImage(
            file=self.relative_to_assets3("image_17.png"))
        self.image_17 = self.canvas.create_image(
            522.0,
            226.0,
            image=self.image_image_17
        )

        self.image_image_18 = PhotoImage(
            file=self.relative_to_assets3("image_18.png"))
        self.image_18 = self.canvas.create_image(
            495.0,
            96.0,
            image=self.image_image_18
        )

        self.image_image_19 = PhotoImage(
            file=self.relative_to_assets3("image_19.png"))
        self.image_19 = self.canvas.create_image(
            428.0,
            76.0,
            image=self.image_image_19
        )

        self.image_image_20 = PhotoImage(
            file=self.relative_to_assets3("image_20.png"))
        self.image_20 = self.canvas.create_image(
            348.0,
            55.0,
            image=self.image_image_20
        )

        self.image_image_21 = PhotoImage(
            file=self.relative_to_assets3("image_21.png"))
        self.image_21 = self.canvas.create_image(
            637.0,
            51.0,
            image=self.image_image_21
        )

        self.initial_frame(source)

    def validate_input(self, value):
        return value.isdigit() or value == ""

    def initial_frame(self, source):
        self.width = int(self.entry_1.get())
        self.inx1 = int(self.entry_2.get())
        self.outx1 = int(self.entry_3.get())
        self.iny1 = int(self.entry_4.get())
        self.outy1 = int(self.entry_5.get())
        self.inx2 = int(self.entry_6.get())
        self.outx2 = int(self.entry_7.get())
        self.iny2 = int(self.entry_8.get())
        self.outy2 = int(self.entry_9.get())
        self.height = int(self.entry_10.get())

        cap = cv2.VideoCapture(source)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (self.width, self.height))

        # table
        data = [
            ("BUS", 'X', 'X', ('X' + 'X')),
            ("JEEP", 'X', 'X', ('X' + 'X')),
            ("MCYCLE", 'X', 'X', ('X' + 'X')),
            ("TCYCLE", 'X', 'X', ('X' + 'X')),
            ("VAN", 'X', 'X', ('X' + 'X')),
            ("CAR", 'X', 'X', ('X' + 'X')),
            ("TRUCK", 'X', 'X', ('X' + 'X')),
            ("TOTAL", 'X', 'X', ('X' + 'X')),
        ]

        # Draw the table header
        cv2.putText(frame, "Class",
                    (x_start, y_start),
                    font, font_scale, font_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "IN",
                    (x_start + col_width, y_start),
                    font, font_scale, font_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "OUT",
                    (x_start + 2 * col_width, y_start),
                    font, font_scale, font_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "TTL",
                    (x_start + 3 * col_width, y_start),
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Draw the table data
        for i, (class_name, in_count, out_count, total_count) in enumerate(data, start=1):
            y_position = y_start + i * line_height
            cv2.putText(frame, class_name, (x_start, y_position),
                        font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, str(in_count), (x_start + col_width, y_position),
                        font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, str(out_count), (x_start + 2 * col_width, y_position),
                        font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, str(total_count), (x_start + 3 * col_width, y_position),
                        font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        cv2.putText(frame, "IN", (self.inx2 // 2, self.iny1 - 5),
                    font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (self.inx1, self.iny1), (self.inx2, self.iny2), (255, 165, 0), 2)
        cv2.putText(frame, "CHECK", (self.inx2 // 2, self.iny1 + 15),
                    font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "OUT", (self.outx2 // 2, self.outy1 - 5),
                    font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (self.outx1, self.outy1), (self.outx2, self.outy2), (255, 255, 0), 2)
        cv2.putText(frame, "CHECK", (self.outx2 // 2, self.outy1 + 15),
                    font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Check Frame", frame)

    def check_frame(self, source):
        global tip_count
        self.width = int(self.entry_1.get())
        self.inx1 = int(self.entry_2.get())
        self.outx1 = int(self.entry_3.get())
        self.iny1 = int(self.entry_4.get())
        self.outy1 = int(self.entry_5.get())
        self.inx2 = int(self.entry_6.get())
        self.outx2 = int(self.entry_7.get())
        self.iny2 = int(self.entry_8.get())
        self.outy2 = int(self.entry_9.get())
        self.height = int(self.entry_10.get())

        cap = cv2.VideoCapture(source)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (self.width, self.height))

        # table
        data = [
            ("BUS", 'X', 'X', ('X' + 'X')),
            ("JEEP", 'X', 'X', ('X' + 'X')),
            ("MCYCLE", 'X', 'X', ('X' + 'X')),
            ("TCYCLE", 'X', 'X', ('X' + 'X')),
            ("VAN", 'X', 'X', ('X' + 'X')),
            ("CAR", 'X', 'X', ('X' + 'X')),
            ("TRUCK", 'X', 'X', ('X' + 'X')),
            ("TOTAL", 'X', 'X', ('X' + 'X')),
        ]

        # Draw the table header
        cv2.putText(frame, "Class",
                    (x_start, y_start),
                    font, font_scale, font_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "IN",
                    (x_start + col_width, y_start),
                    font, font_scale, font_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "OUT",
                    (x_start + 2 * col_width, y_start),
                    font, font_scale, font_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "TTL",
                    (x_start + 3 * col_width, y_start),
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Draw the table data
        for i, (class_name, in_count, out_count, total_count) in enumerate(data, start=1):
            y_position = y_start + i * line_height
            cv2.putText(frame, class_name, (x_start, y_position),
                        font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, str(in_count), (x_start + col_width, y_position),
                        font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, str(out_count), (x_start + 2 * col_width, y_position),
                        font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, str(total_count), (x_start + 3 * col_width, y_position),
                        font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        cv2.putText(frame, "IN", (self.inx2 // 2, self.iny1 - 5),
                    font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (self.inx1, self.iny1), (self.inx2, self.iny2), (255, 165, 0), 2)
        cv2.putText(frame, "CHECK", (self.inx2 // 2, self.iny1 + 15),
                    font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "OUT", (self.outx2 // 2, self.outy1 - 5),
                    font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (self.outx1, self.outy1), (self.outx2, self.outy2), (255, 255, 0), 2)
        cv2.putText(frame, "CHECK", (self.outx2 // 2, self.outy1 + 15),
                    font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Check Frame", frame)

    def relative_to_assets3(self, path: str) -> Path:
        return modification_asset_path / Path(path)

    def model_predict(self, mode):
        self.model_predict_frame.destroy()
        self.model_predict_frame = Frame(self.master, bg="#000E1D")
        self.model_predict_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = Canvas(
            self.model_predict_frame,
            bg="#000E1D",
            height=550,
            width=700,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets4("image_1.png"))
        self.image_1 = self.canvas.create_image(
            350.0,
            275.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets4("image_2.png"))
        self.image_2 = self.canvas.create_image(
            503.0,
            275.0,
            image=self.image_image_2
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets4("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.print_csv_in_content(),
            relief="flat"
        )
        self.button_1.place(
            x=292.0,
            y=403.0,
            width=408.0,
            height=56.0
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets4("image_3.png"))
        self.image_3 = self.canvas.create_image(
            356.0,
            151.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets4("image_4.png"))
        self.image_4 = self.canvas.create_image(
            495.0,
            96.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(
            file=self.relative_to_assets4("image_5.png"))
        self.image_5 = self.canvas.create_image(
            428.0,
            76.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=self.relative_to_assets4("image_6.png"))
        self.image_6 = self.canvas.create_image(
            348.0,
            55.0,
            image=self.image_image_6
        )

        self.image_image_7 = PhotoImage(
            file=self.relative_to_assets4("image_7.png"))
        self.image_7 = self.canvas.create_image(
            511.0,
            255.0,
            image=self.image_image_7
        )

        self.image_image_8 = PhotoImage(
            file=self.relative_to_assets4("image_8.png"))
        self.image_8 = self.canvas.create_image(
            643.0,
            52.0,
            image=self.image_image_8
        )

        model = YOLO(model8)
        my_file = open("labels.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")

        cap = cv2.VideoCapture(mode)

        # initialize
        count = 0
        vh_up = {}
        count_up = []
        vh_down = {}
        count_down = []
        tracker = Tracker()
        v_counts = {
            'Bus': {'in': 0, 'out': 0},
            'Jeep': {'in': 0, 'out': 0},
            'Motorcycle': {'in': 0, 'out': 0},
            'Tricycle': {'in': 0, 'out': 0},
            'Van': {'in': 0, 'out': 0},
            'Car': {'in': 0, 'out': 0},
            'Truck': {'in': 0, 'out': 0},
        }

        yellow = (255, 255, 0, 255)
        offset = 6

        start_time = datetime.datetime.now().strftime("%H:%M:%S")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % 3 != 0:
                continue
            frame = cv2.resize(frame, (self.width, self.height))

            results = model.track(frame)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            list = []

            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                c = class_list[int(row[len(row) - 1])]
                list.append([x1, y1, x2, y2, c])
            print(f"bbox = {list}")
            bbox_id = tracker.update(list)
            for bbox in bbox_id:
                x3, y3, x4, y4, id, c_name = bbox
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2
                cv2.rectangle(frame, (x3, y3), (x4, y4), yellow, 2)
                cv2.putText(frame, f"{id} {c_name}",
                            (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (b, g, r), 2, cv2.LINE_AA)

                # going down
                if (cy + offset) > self.iny1 > (cy - offset) and (cy + offset) > self.iny2 > (cy - offset):
                    vh_down[id] = cy
                if id in vh_down and id not in vh_up:
                    if count_down.count(id) == 0:
                        count_down.append(id)
                        v_counts[c_name]['in'] += 1

                # going up
                if (cy + offset) > self.outy1 > (cy - offset) and (cy + offset) > self.outy2 > (cy - offset):
                    vh_up[id] = cy
                if id in vh_up and id not in vh_down:
                    if count_up.count(id) == 0:
                        count_up.append(id)
                        v_counts[c_name]['out'] += 1

            # table
            data = [
                ("BUS", v_counts['Bus']['in'], v_counts['Bus']['out'], (v_counts['Bus']['in'] + v_counts['Bus']['out'])),
                ("JEEP", v_counts['Jeep']['in'], v_counts['Jeep']['out'], (v_counts['Jeep']['in'] + v_counts['Jeep']['out'])),
                ("MCYCLE", v_counts['Motorcycle']['in'], v_counts['Motorcycle']['out'], (v_counts['Motorcycle']['in'] + v_counts['Motorcycle']['out'])),
                ("TCYCLE", v_counts['Tricycle']['in'], v_counts['Tricycle']['out'], (v_counts['Tricycle']['in'] + v_counts['Tricycle']['out'])),
                ("VAN", v_counts['Van']['in'], v_counts['Van']['out'], (v_counts['Van']['in'] + v_counts['Van']['out'])),
                ("CAR", v_counts['Car']['in'], v_counts['Car']['out'], (v_counts['Car']['in'] + v_counts['Car']['out'])),
                ("TRUCK", v_counts['Truck']['in'], v_counts['Truck']['out'], (v_counts['Truck']['in'] + v_counts['Truck']['out'])),
                ("TOTAL", len(count_down), len(count_up), (len(count_down) + len(count_up)))
            ]

            # Draw the table header
            cv2.putText(frame, "Class",
                        (x_start, y_start),
                        font, font_scale, font_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "IN",
                        (x_start + col_width, y_start),
                        font, font_scale, font_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "OUT",
                        (x_start + 2 * col_width, y_start),
                        font, font_scale, font_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "TTL",
                        (x_start + 3 * col_width, y_start),
                        font, font_scale, font_color, 2, cv2.LINE_AA)

            # Draw the table data
            for i, (class_name, in_count, out_count, total_count) in enumerate(data, start=1):
                y_position = y_start + i * line_height
                cv2.putText(frame, class_name, (x_start, y_position),
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(frame, str(in_count), (x_start + col_width, y_position),
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(frame, str(out_count), (x_start + 2 * col_width, y_position),
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(frame, str(total_count), (x_start + 3 * col_width, y_position),
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            cv2.putText(frame, "IN", (self.inx2 // 2, self.iny1 - 5),
                        font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (self.inx1, self.iny1), (self.inx2, self.iny2), (255, 165, 0), 2)
            cv2.putText(frame, "CHECK", (self.inx2 // 2, self.iny1 + 15),
                        font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, "OUT", (self.outx2 // 2, self.outy1 - 5),
                        font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (self.outx1, self.outy1), (self.outx2, self.outy2), (255, 255, 0), 2)
            cv2.putText(frame, "CHECK", (self.outx2 // 2, self.outy1 + 15),
                        font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

            print(f"Down = {count_down} \nUp = {count_up}")
            cv2.imshow("VCount[ing]", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        end_time = datetime.datetime.now().strftime("%H:%M:%S")
        cap.release()
        cv2.destroyAllWindows()
        self.write_to_csv(start_time, v_counts, count_up, count_down, end_time)

    def write_to_csv(self, start_time, v_counts, count_up, count_down, end_time):

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        in_data = [current_date,
                   start_time,
                   v_counts['Bus']['in'],
                   v_counts['Jeep']['in'],
                   v_counts['Motorcycle']['in'],
                   v_counts['Tricycle']['in'],
                   v_counts['Van']['in'],
                   v_counts['Car']['in'],
                   v_counts['Truck']['in'],
                   len(count_down),
                   end_time]
        file_exists = os.path.isfile(in_csv_file_path)
        with open(in_csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                                 'IN_TOTAL' 'END_TIME'])
            writer.writerow(in_data)

        out_data = [current_date,
                    start_time,
                    v_counts['Bus']['out'],
                    v_counts['Jeep']['out'],
                    v_counts['Motorcycle']['out'],
                    v_counts['Tricycle']['out'],
                    v_counts['Van']['out'],
                    v_counts['Car']['out'],
                    v_counts['Truck']['out'],
                    len(count_up),
                    end_time]
        file_exists = os.path.isfile(out_csv_file_path)
        with open(out_csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                                 'OUT_TOTAL' 'END_TIME'])
            writer.writerow(out_data)

        all_data = [current_date,
                    start_time,
                    (v_counts['Bus']['in'] + v_counts['Bus']['out']),
                    (v_counts['Jeep']['in'] + v_counts['Jeep']['out']),
                    (v_counts['Motorcycle']['in'] + v_counts['Motorcycle']['out']),
                    (v_counts['Tricycle']['in'] + v_counts['Tricycle']['out']),
                    (v_counts['Van']['in'] + v_counts['Van']['out']),
                    (v_counts['Car']['in'] + v_counts['Car']['out']),
                    (v_counts['Truck']['in'] + v_counts['Truck']['out']),
                    len(count_down),
                    len(count_up),
                    (len(count_down) + len(count_up)),
                    end_time]
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                                 'IN_COUNT', 'OUT_COUNT', 'TOTAL', 'END_TIME'])
            writer.writerow(all_data)

    def relative_to_assets4(self, path: str) -> Path:
        return predict_note_asset_path / Path(path)

    def print_csv_in_content(self):
        self.print_csv_in_content_frame.destroy()
        self.print_csv_in_content_frame = Frame(self.master, bg="#000E1D")
        self.print_csv_in_content_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = Canvas(
            self.print_csv_in_content_frame,
            bg="#000E1D",
            height=550,
            width=700,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets5("image_1.png"))
        self.image_1 = self.canvas.create_image(
            350.0,
            275.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets5("image_2.png"))
        self.image_2 = self.canvas.create_image(
            503.0,
            275.0,
            image=self.image_image_2
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets5("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.print_csv_out_content(),
            relief="flat"
        )
        self.button_1.place(
            x=292.0,
            y=403.0,
            width=408.0,
            height=56.0
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets5("image_3.png"))
        self.image_3 = self.canvas.create_image(
            390.0,
            151.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets5("image_4.png"))
        self.image_4 = self.canvas.create_image(
            495.0,
            96.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(
            file=self.relative_to_assets5("image_5.png"))
        self.image_5 = self.canvas.create_image(
            428.0,
            76.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=self.relative_to_assets5("image_6.png"))
        self.image_6 = self.canvas.create_image(
            348.0,
            55.0,
            image=self.image_image_6
        )

        self.image_image_7 = PhotoImage(
            file=self.relative_to_assets5("image_7.png"))
        self.image_7 = self.canvas.create_image(
            645.0,
            52.0,
            image=self.image_image_7
        )

        self.in_treeview = ttk.Treeview(self.print_csv_in_content_frame, selectmode="browse", show="headings")
        columns = ['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                   'IN_TOTAL', 'END_TIME']
        self.in_treeview["columns"] = columns
        for col in columns:
            self.in_treeview.heading(col, text=col)
        with open(in_csv_file_path, mode="r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.in_treeview.insert("", "end", values=row)

        self.vertical_scrollbar = Scrollbar(self.print_csv_in_content_frame,
                                            command=self.in_treeview.yview)
        self.vertical_scrollbar.place(x=660, y=180, height=190)

        self.horizontal_scrollbar = Scrollbar(self.print_csv_in_content_frame,
                                              orient="horizontal",
                                              command=self.in_treeview.xview)
        self.horizontal_scrollbar.place(x=310, y=369, width=350)

        self.in_treeview.configure(yscrollcommand=self.vertical_scrollbar.set,
                                   xscrollcommand=self.horizontal_scrollbar.set)
        self.in_treeview.place(x=310, y=180, width=350, height=190)

    def relative_to_assets5(self, path: str) -> Path:
        return in_result_asset_path / Path(path)

    def print_csv_out_content(self):
        self.print_csv_out_content_frame.destroy()
        self.print_csv_out_content_frame = Frame(self.master, bg="#000E1D")
        self.print_csv_out_content_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = Canvas(
            self.print_csv_out_content_frame,
            bg="#000E1D",
            height=550,
            width=700,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets6("image_1.png"))
        self.image_1 = self.canvas.create_image(
            350.0,
            275.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets6("image_2.png"))
        self.image_2 = self.canvas.create_image(
            503.0,
            275.0,
            image=self.image_image_2
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets6("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.print_csv_all_content(),
            relief="flat"
        )
        self.button_1.place(
            x=292.0,
            y=403.0,
            width=408.0,
            height=56.0
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets6("image_3.png"))
        self.image_3 = self.canvas.create_image(
            407.0,
            151.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets6("image_4.png"))
        self.image_4 = self.canvas.create_image(
            495.0,
            96.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(
            file=self.relative_to_assets6("image_5.png"))
        self.image_5 = self.canvas.create_image(
            428.0,
            76.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=self.relative_to_assets6("image_6.png"))
        self.image_6 = self.canvas.create_image(
            348.0,
            55.0,
            image=self.image_image_6
        )

        self.image_image_7 = PhotoImage(
            file=self.relative_to_assets6("image_7.png"))
        self.image_7 = self.canvas.create_image(
            645.0,
            52.0,
            image=self.image_image_7
        )

        self.out_treeview = ttk.Treeview(self.print_csv_out_content_frame, selectmode="browse", show="headings")
        columns = ['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                   'OUT_TOTAL', 'END_TIME']
        self.out_treeview["columns"] = columns
        for col in columns:
            self.out_treeview.heading(col, text=col)
        with open(out_csv_file_path, mode="r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.out_treeview.insert("", "end", values=row)

        self.vertical_scrollbar = Scrollbar(self.print_csv_out_content_frame,
                                            command=self.out_treeview.yview)
        self.vertical_scrollbar.place(x=660, y=180, height=190)

        self.horizontal_scrollbar = Scrollbar(self.print_csv_out_content_frame,
                                              orient="horizontal",
                                              command=self.out_treeview.xview)
        self.horizontal_scrollbar.place(x=310, y=369, width=350)

        self.out_treeview.configure(yscrollcommand=self.vertical_scrollbar.set,
                                    xscrollcommand=self.horizontal_scrollbar.set)
        self.out_treeview.place(x=310, y=180, width=350, height=190)

    def relative_to_assets6(self, path: str) -> Path:
        return out_result_asset_path / Path(path)

    def print_csv_all_content(self):
        self.print_csv_all_content_frame.destroy()
        self.print_csv_all_content_frame = Frame(self.master, bg="#000E1D")
        self.print_csv_all_content_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = Canvas(
            self.print_csv_all_content_frame,
            bg="#000E1D",
            height=550,
            width=700,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets7("image_1.png"))
        self.image_1 = self.canvas.create_image(
            350.0,
            275.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets7("image_2.png"))
        self.image_2 = self.canvas.create_image(
            503.0,
            275.0,
            image=self.image_image_2
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets7("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.exit_page(),
            relief="flat"
        )
        self.button_1.place(
            x=292.0,
            y=403.0,
            width=408.0,
            height=56.0
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets7("image_3.png"))
        self.image_3 = self.canvas.create_image(
            400.0,
            151.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets7("image_4.png"))
        self.image_4 = self.canvas.create_image(
            495.0,
            96.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(
            file=self.relative_to_assets7("image_5.png"))
        self.image_5 = self.canvas.create_image(
            428.0,
            76.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=self.relative_to_assets7("image_6.png"))
        self.image_6 = self.canvas.create_image(
            348.0,
            55.0,
            image=self.image_image_6
        )

        self.image_image_7 = PhotoImage(
            file=self.relative_to_assets7("image_7.png"))
        self.image_7 = self.canvas.create_image(
            645.0,
            52.0,
            image=self.image_image_7
        )

        self.all_treeview = ttk.Treeview(self.print_csv_all_content_frame, selectmode="browse", show="headings")
        columns = ['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                   'IN_COUNT', 'OUT_COUNT', 'TOTAL', 'END_TIME']
        self.all_treeview["columns"] = columns
        for col in columns:
            self.all_treeview.heading(col, text=col)
        with open(csv_file_path, mode="r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.all_treeview.insert("", "end", values=row)

        self.vertical_scrollbar = Scrollbar(self.print_csv_all_content_frame,
                                            command=self.all_treeview.yview)
        self.vertical_scrollbar.place(x=660, y=180, height=190)

        self.horizontal_scrollbar = Scrollbar(self.print_csv_all_content_frame,
                                              orient="horizontal",
                                              command=self.all_treeview.xview)
        self.horizontal_scrollbar.place(x=310, y=369, width=350)

        self.all_treeview.configure(yscrollcommand=self.vertical_scrollbar.set,
                                    xscrollcommand=self.horizontal_scrollbar.set)
        self.all_treeview.place(x=310, y=180, width=350, height=190)

    def relative_to_assets7(self, path: str) -> Path:
        return all_result_asset_path / Path(path)

    def exit_page(self):
        self.exit_page_frame.destroy()
        self.exit_page_frame = Frame(self.master, bg="#000E1D")
        self.exit_page_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = Canvas(
            self.exit_page_frame,
            bg="#000E1D",
            height=550,
            width=700,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets8("image_1.png"))
        self.image_1 = self.canvas.create_image(
            350.0,
            275.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=self.relative_to_assets8("image_2.png"))
        self.image_2 = self.canvas.create_image(
            503.0,
            275.0,
            image=self.image_image_2
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets8("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: exit(0),
            relief="flat"
        )
        self.button_1.place(
            x=292.0,
            y=328.0,
            width=408.0,
            height=56.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets8("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.home_page(),
            relief="flat"
        )
        self.button_2.place(
            x=292.0,
            y=394.0,
            width=408.0,
            height=56.0
        )

        self.image_image_3 = PhotoImage(
            file=self.relative_to_assets8("image_3.png"))
        self.image_3 = self.canvas.create_image(
            416.0,
            151.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=self.relative_to_assets8("image_4.png"))
        self.image_4 = self.canvas.create_image(
            495.0,
            96.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(
            file=self.relative_to_assets8("image_5.png"))
        self.image_5 = self.canvas.create_image(
            428.0,
            76.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=self.relative_to_assets8("image_6.png"))
        self.image_6 = self.canvas.create_image(
            348.0,
            55.0,
            image=self.image_image_6
        )

        self.image_image_7 = PhotoImage(
            file=self.relative_to_assets8("image_7.png"))
        self.image_7 = self.canvas.create_image(
            644.0,
            52.0,
            image=self.image_image_7
        )

    def relative_to_assets8(self, path: str) -> Path:
        return end_page_asset_path / Path(path)

if __name__ == '__main__':
    model7 = ".\\model_results\\train4\\weights\\best.pt"
    model8 = ".\\model_results\\train4b\\weights\\best.pt" # the best
    model9 = ".\\model_results\\train4c\\weights\\best.pt"
    model10 = ".\\model_results\\train5a\\weights\\best.pt"

    # PATHS to frames
    OUTPUT_PATH = Path(__file__).parent
    home_asset_path = OUTPUT_PATH / Path(r".\build\assets\frame0")
    select_video_asset_path = OUTPUT_PATH / Path(r".\build\assets\frame1")
    select_camera_asset_path = OUTPUT_PATH / Path(r".\build\assets\frame2")
    modification_asset_path = OUTPUT_PATH / Path(r".\build\assets\frame3")
    predict_note_asset_path = OUTPUT_PATH / Path(r".\build\assets\frame4")
    in_result_asset_path = OUTPUT_PATH / Path(r".\build\assets\frame5")
    out_result_asset_path = OUTPUT_PATH / Path(r".\build\assets\frame7")
    all_result_asset_path = OUTPUT_PATH / Path(r".\build\assets\frame8")
    end_page_asset_path = OUTPUT_PATH / Path(r".\build\assets\frame6")

    # PATHS to csv
    csv_file_path = "./csv_results/allRESULTS.csv"
    in_csv_file_path = "./csv_results/inRESULTS.csv"
    out_csv_file_path = "./csv_results/outRESULTS.csv"

    # color
    b, g, r = 0, 255, 0

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (0,255,0)

    # Text positions
    x_start = 10
    y_start = 30
    line_height = 20
    col_width = 80

    root = Tk()
    root.title("VCount")
    app = count_vehicles(root)
    root.mainloop()
