
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, messagebox, Listbox
from pygrabber.dshow_graph import FilterGraph


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"D:\Downloads\Tkinter-Designer-master\model9-gui-d\build\assets\frame2")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def refresh_devices():
    device_listbox.delete(0, "end")
    cameras = get_camera_devices()
    for idx, info in enumerate(cameras, start=1):
        device_listbox.insert("end", f"{info}")

def get_camera_devices():
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

def get_camera_input():
    selected_index = device_listbox.curselection()
    if selected_index:
        devices = get_camera_devices()
        if 0 <= int(selected_index[0]) <= len(devices):
            print(int(selected_index[0]))
        else:
            messagebox.showwarning("Invalid Index", "Please enter a valid camera device index.")
    else:
        messagebox.showwarning("No Camera Selected", "Please select a camera.")


window = Tk()

window.geometry("700x550")
window.configure(bg = "#000E1D")


canvas = Canvas(
    window,
    bg = "#000E1D",
    height = 550,
    width = 700,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    350.0,
    275.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    503.0,
    275.0,
    image=image_image_2
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=292.0,
    y=377.0,
    width=408.0,
    height=56.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=292.0,
    y=444.0,
    width=408.0,
    height=56.0
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    441.0,
    179.0,
    image=image_image_3
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    495.0,
    96.0,
    image=image_image_4
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    428.0,
    76.0,
    image=image_image_5
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    351.0,
    91.0,
    image=image_image_6
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    641.0,
    52.0,
    image=image_image_7
)

device_listbox = Listbox(selectmode="single", justify="center")
device_listbox.place(x=313.0, y=200.0, width=360.0, height=130.0)
scrollbar = Scrollbar(command=device_listbox.yview)
scrollbar.place(x=660.0, y=200.0, height=130.0)
device_listbox.config(yscrollcommand=scrollbar.set)

refresh_button = Button(text="Refresh", command=refresh_devices)
refresh_button.place(x=500.0, y=340.0, width=170.0, height=30)

refresh_devices()

window.resizable(False, False)
window.mainloop()