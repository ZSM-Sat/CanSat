#!/usr/bin/env python
import sys
import time
from SX127x.LoRa import *
from SX127x.LoRaArgumentParser import LoRaArgumentParser
from SX127x.board_config import BOARD
import RPi.GPIO as GPIO
import board
import adafruit_bmp280
import w1thermsensor
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
from threading import Thread, Event

event = Event()

# Setup variables used between threads
global max_conf
max_conf = 0
global frame
global tx_counter
tx_counter = 0

# Set colors variables
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Setup camera
camera = PiCamera()
rawCapture = PiRGBArray(camera)
camera.resolution = (384,384)

# Setup buzzer
buzzer = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer, GPIO.OUT)

# Setup pressure and temperature sensors
i2c = board.I2C()  # uses board.SCL and board.SDA
bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(i2c, address=0x76)
sensor = w1thermsensor.W1ThermSensor()
BOARD.setup()

# Setup YOLOv4-tiny model with cv2
names = ['Tree']
print(bcolors.OKCYAN + "[AI]Ladowanie modelu")
net = cv2.dnn.readNet(f'yolo-tiny-obj_best.weights', 'yolo-tiny-obj.cfg')
print(bcolors.OKCYAN + "[AI]Ustawienia")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print(bcolors.OKCYAN + "[AI]Detection model")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(384, 384), scale=1/255, swapRB=True)


# Our CanSat main class
class LoRaBeacon(LoRa):
    def __init__(self, verbose=False):
        super(LoRaBeacon, self).__init__(verbose)
        self.set_mode(MODE.SLEEP)
        self.set_dio_mapping([1,0,0,0,0,0])

    def on_rx_done(self):
        print(bcolors.OKGREEN + "[INFO] Odebrany Pakiet?")

    # Increments tx_counter on successfully sent packet and sends audio signal with buzzer
    def on_tx_done(self):
        global tx_counter
        self.set_mode(MODE.STDBY)
        self.clear_irq_flags(TxDone=1)
        print(bcolors.OKGREEN + "[INFO]Pakiet wyslany: #%d" % tx_counter)
        tx_counter += 1
        GPIO.output(buzzer, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(buzzer, GPIO.LOW)

    def on_cad_done(self):
        print("\non_CadDone")
        print(self.get_irq_flags())

    def on_rx_timeout(self):
        print("\non_RxTimeout")
        print(self.get_irq_flags())

    def on_valid_header(self):
        print("\non_ValidHeader")
        print(self.get_irq_flags())

    def on_payload_crc_error(self):
        print("\non_PayloadCrcError")
        print(self.get_irq_flags())

    def on_fhss_change_channel(self):
        print("\non_FhssChangeChannel")
        print(self.get_irq_flags())

    # Starts sending telemetry
    def start(self):
        global tx_counter
        global frame
        global max_conf

        # Takes first photo and saves it
        print(bcolors.OKGREEN + "[INFO]Pierwsze zdjecie")
        camera.capture(rawCapture, format="bgr")
        frame = rawCapture.array
        print(bcolors.OKGREEN + "[INFO]Zapisanie na karcie")
        cv2.imwrite('/home/pi/photos/picture_pierwsze.jpg', frame)
        
        # Sets bmp280 sea level pressure to actual pressure
        bmp280.sea_level_pressure = bmp280.pressure

        # Creates and starts AI thread
        t = Thread(target=photoai, args=())
        t.start()

        # In loop takes photos, saves them, takes sensors values and sends telemetry to ground station as a string
        # "$ZSMSAT:packet id;temperature;bmp temperature;bmp pressure;bmp altitude;max confidence in trees detection;END"
        while True:
            rawCapture.truncate(0)
            camera.capture(rawCapture, format="bgr")
            frame = rawCapture.array
            cv2.imwrite('/home/pi/photos/picture_'+ str(tx_counter) +'.jpg', frame)
            bmp_temp = round(bmp280.temperature,2)
            bmp_pres = round(bmp280.pressure,2)
            bmp_alti = round(bmp280.altitude,2)
            temp = round(sensor.get_temperature(),2)
            payload = "$ZSMSAT:" + str(tx_counter) + ";" + str(temp) + ";" + str(bmp_temp) + ";" + str(bmp_pres) + ";" + str(bmp_alti) + ";" + str(round(max_conf,2)) + ";END"
            self.write_payload(list(payload.encode('ascii')))
            self.set_mode(MODE.TX)
            print(bcolors.WARNING + "[PAYLOAD] " + payload)


# Trees detection thread function
def photoai():
    global tx_counter
    global frame
    global max_conf

    # In a loop gets current photo from camera, detects trees, marks results on image and saves this image
    while True:
        id2 = tx_counter
        print(bcolors.HEADER + "[PHOTO_THREAD]Wczytany obraz nr: " + str(id2))
        frame2 = frame
        print(bcolors.HEADER + "[PHOTO_THREAD]Wykrywanie...")
        tic = time.perf_counter()
        classes, confidences, boxes = model.detect(frame2, confThreshold=0.6, nmsThreshold=0.4)
        toc = time.perf_counter()
        print(bcolors.HEADER + "[PHOTO_THREAD]Oznaczanie...")

        if len(classes) > 0:
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%.2f' % confidence
                label = '%s: %s' % (names[classId], label)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                left, top, width, height = box
                top = max(top, labelSize[1])
                cv2.rectangle(frame2, box, color=(0, 255, 0), thickness=3)
                cv2.rectangle(frame2, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame2, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                if max_conf < confidence:
                    max_conf = confidence
            print(bcolors.HEADER + "[PHOTO_THREAD]Wykryto " + str(max_conf))
            cv2.imwrite('/home/pi/photos/picture_analized_' + str(id2) + '.jpg', frame2)
        else:
            print("[PHOTO_THREAD]Nic nie wykryto!")

print(bcolors.OKGREEN + "[INFO]Ustawienia Lora")

# Create main object and setup radio communication
lora = LoRaBeacon(verbose=False)

lora.set_pa_config(pa_select=1)
#lora.set_rx_crc(True)
lora.set_agc_auto_on(True)
#lora.set_lna_gain(GAIN.NOT_USED)
lora.set_coding_rate(CODING_RATE.CR4_5)
lora.set_implicit_header_mode(False)
lora.set_freq(433)
lora.set_spreading_factor(9)

#lora.set_pa_config(max_power=0x04, output_power=0x0F)
#lora.set_pa_config(max_power=0x04, output_power=0b01000000)
lora.set_low_data_rate_optim(True)
#lora.set_pa_ramp(PA_RAMP.RAMP_50_us)

#assert(lora.get_lna()['lna_gain'] == GAIN.NOT_USED)
assert(lora.get_agc_auto_on() == 1)

# Starts our CanSat script and handles exiting by Ctrl-C
try:
    lora.start()
except KeyboardInterrupt:
    event.set()
    sys.stdout.flush()
    print("")
    sys.stderr.write("KeyboardInterrupt\n")
finally:
    sys.stdout.flush()
    print("")
    lora.set_mode(MODE.SLEEP)
    print(lora)
    BOARD.teardown()
    GPIO.cleanup()
