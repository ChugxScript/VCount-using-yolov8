from ultralytics import YOLO
import cv2
import os
import datetime
import csv
import pandas as pd
from tracker import*


# TRAIN, VALIDATION, AND TESTING OF THE DATA AND MODEL
def create_model():
    model = YOLO("yolov8n.pt")  # build a new model from scratch
    model.train(data="data.yaml", epochs=100, batch=16, imgsz=640, project='./model_results', name='train3.0')
    model.val(data='data.yaml', project='./model_results', name='val4.0')
    model.predict('./test3.0/images', save=True, save_txt=True, conf=0.7, project='./model_results', name='test3.0')


# FOR USER INTERFACE
class count_vehicles:
    def __init__(self):
        csv_file_path = "./csv_results/RESULTS.csv"
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                                 'IN_COUNT', 'OUT_COUNT', 'TOTAL', 'END_TIME'])

        self.home_page()

    def home_page(self):
        print("Count Vehicles")
        print("Press any key to continue")
        input()
        self.select_options()

    def select_options(self):
        print("[1] - Process Video file")
        print("[2] - Select Camera Devices")
        user_num = int(input(print("-> ")))

        self.print_csv_content()
        if user_num == 1:
            self.get_video_file()
        else:
            self.select_camera_device()

        self.exit_page()

    # IF THE USER CHOOSE TO LOCATE FILES OR VID
    def get_video_file(self):
        self.model_predict(vid_test_tbus)

    # IF THE USER WANT TO USE CAMERA OR LIVE CAM TRACKING
    def select_camera_device(self):
        self.model_predict(1)

    def model_predict(self, mode):
        model = YOLO(model8)
        my_file = open("labels.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")

        def RGB(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                colorsBGR = [x, y]
                print(colorsBGR)

        cv2.namedWindow('RGB')
        cv2.setMouseCallback('RGB', RGB)

        cap = cv2.VideoCapture(mode)

        # initialize
        count = 0
        vh_up = {}
        count_up = []
        vh_down = {}
        count_down = []
        c_bus = 0
        c_jeep = 0
        c_motor = 0
        c_tric = 0
        c_van = 0
        c_car = 0
        c_truck = 0
        tracker = Tracker()

        # colors
        yellow = (255, 255, 0, 255)
        b, g, r = 0, 255, 0

        # set the line coordinates
        cy1 = 235
        cy2 = 368
        offset = 6

        start_time = datetime.datetime.now().strftime("%H:%M:%S")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % 3 != 0:
                continue
            frame = cv2.resize(frame, (1020, 500))
            # frame = cv2.resize(frame, (1920, 1080))

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
                if cy1 < (cy+offset) and cy1 > (cy-offset):
                    vh_down[id] = cy
                if id in vh_down and id not in vh_up:
                    if count_down.count(id) == 0:
                        count_down.append(id)
                        if c_name == 'Bus':
                            c_bus += 1
                        elif c_name == 'Jeep':
                            c_jeep += 1
                        elif c_name == 'Motorcycle':
                            c_motor += 1
                        elif c_name == 'Tricycle':
                            c_tric += 1
                        elif c_name == 'Van':
                            c_van += 1
                        elif c_name == 'Car':
                            c_car += 1
                        elif c_name == 'Truck':
                            c_truck += 1

                # going up
                if cy2 < (cy + offset) and cy2 > (cy - offset):
                    vh_up[id] = cy
                if id in vh_up and id not in vh_down:
                    if count_up.count(id) == 0:
                        count_up.append(id)
                        if c_name == 'Bus':
                            c_bus += 1
                        elif c_name == 'Jeep':
                            c_jeep += 1
                        elif c_name == 'Motorcycle':
                            c_motor += 1
                        elif c_name == 'Tricycle':
                            c_tric += 1
                        elif c_name == 'Van':
                            c_van += 1
                        elif c_name == 'Car':
                            c_car += 1
                        elif c_name == 'Truck':
                            c_truck += 1

            # line
            cv2.putText(frame, f"BUS = {c_bus}",
                        (60, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (b, g, r), 1, cv2.LINE_AA)

            cv2.putText(frame, f"JEEP = {c_jeep}",
                        (60, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (b, g, r), 1, cv2.LINE_AA)

            cv2.putText(frame, f"MOTORCYCLE = {c_motor}",
                        (60, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (b, g, r), 1, cv2.LINE_AA)

            cv2.putText(frame, f"TRICYCLE = {c_tric}",
                        (60, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (b, g, r), 1, cv2.LINE_AA)

            cv2.putText(frame, f"VAN = {c_van}",
                        (60, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (b, g, r), 1, cv2.LINE_AA)

            cv2.putText(frame, f"CAR = {c_car}",
                        (60, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (b, g, r), 1, cv2.LINE_AA)

            cv2.putText(frame, f"TRUCK = {c_truck}",
                        (60, 170), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (b, g, r), 1, cv2.LINE_AA)

            cv2.line(frame, (0, cy1), (1020, cy1), (255, 255, 255), 1)
            cv2.putText(frame, 'IN = ' + str(len(count_up)),
                        (60, 260), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (b, g, r), 1, cv2.LINE_AA)

            cv2.line(frame, (0, cy2), (1020, cy2), (255, 255, 255), 1)
            cv2.putText(frame, 'OUT = ' + str(len(count_down)),
                        (60, 290), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (b, g, r), 1, cv2.LINE_AA)

            cv2.putText(frame, 'TOTAL = ' + str(len(count_down + count_up)),
                        (60, 320), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (b, g, r), 2, cv2.LINE_AA)

            print(f"Down = {count_down} \nUp = {count_up}")
            cv2.imshow("RGB", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        end_time = datetime.datetime.now().strftime("%H:%M:%S")
        cap.release()
        cv2.destroyAllWindows()
        self.write_to_csv(start_time, c_bus, c_jeep, c_motor, c_tric, c_van, c_car, c_truck,
                          len(count_up), len(count_down), len(count_up + count_down), end_time)

    def write_to_csv(self, start_time, cbus, cjeep, cmotor, ctric, cvan, ccar, ctruck,
                     in_count, out_count, ctotal, end_time):
        csv_file_path = "./csv_results/RESULTS.csv"
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        data = [current_date, start_time, cbus, cjeep, cmotor, ctric, cvan, ccar, ctruck,
                in_count, out_count, ctotal, end_time]
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['DATE', 'START_TIME', 'BUS', 'JEEP', 'MOTORCYCLE', 'TRICYCLE', 'VAN', 'CAR', 'TRUCK',
                                 'IN_COUNT', 'OUT_COUNT', 'TOTAL' 'END_TIME'])

            writer.writerow(data)

        self.print_csv_content()

    def print_csv_content(self):
        global csv_list_count
        csv_file_path = "./csv_results/RESULTS.csv"
        with open(csv_file_path, mode="r") as file:
            reader = csv.reader(file)
            for _ in range(csv_list_count):
                next(reader)
            for row in reader:
                print(row)
                csv_list_count += 1

    def exit_page(self):
        print("Thank you for using Count Vehicles")
        print("[1] - Count again")
        print("[2] - Exit")
        if int(input(print("-> "))) == 1:
            self.home_page()
        else:
            exit(0)


if __name__ == '__main__':
    csv_list_count = 0
    # model1 = "E:\\Portable_model\\model_results\\train1.0\\weights\\best.pt"
    model2 = ".\\model_results\\train2.0\\count_vehicles\\weights\\best.pt" # goods
    model3 = ".\\model_results\\train3.1\\best.pt" # kinda
    model4 = ".\\model_results\\train3.2.1\\weights\\best.pt" #kinda
    # model5 = "E:\\Portable_model\\model_results\\train3.4\\weights\\best.pt"
    model6 = ".\\model_results\\train3.5\\weights\\best.pt"
    model7 = ".\\model_results\\train4\\weights\\best.pt"
    model8 = ".\\model_results\\train4b\\weights\\best.pt"
    model9 = ".\\model_results\\train4c\\weights\\best.pt"
    model10 = ".\\model_results\\train5a\\weights\\best.pt"

    vid_test = "E:\\Portable_model\\top_5mins.mp4"
    vid_test_1 = "E:\\Portable_model\\top_1min.mp4"
    vid_test_b = "E:\\Portable_model\\Test_vid_top.mp4"
    vid_test_tip = "C:\\Users\\andre\\Desktop\\dataset3.0\\Dataset4\\tip_vid_whole.mp4"
    vid_test_tbus = "C:\\Users\\andre\\Desktop\\dataset3.0\\Dataset4\\tip_vid_bus.mp4"

    count_vehicles()
