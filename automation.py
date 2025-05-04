import smtplib
from email.mime.text import MIMEText
import logging
import sys
import time
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import scipy.io
import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import serial
import time
import numpy as np
import os
from datetime import datetime, timedelta, timezone

# timestamp = datetime.datetime.now(datetime.timezone.utc) 

def SendQueryMachineTemp(measurement, measure_point, probe_type, timestamp,value) :
    token = "IP7opg_k5MF_gSaFVaCudqQIjyjVj5fQorzxqzC-JAfxYnotppgHrF9a-DNM3YpOuBaNV_d7_S2PBR5CXjn0dw=="
    url = "http://192.168.0.58:8086"
    org = "JYLab3F"
    bucket="FlourAnalysis"
    point = (
        Point(measurement)
        .tag("Location",measure_point)
        .tag("Type",probe_type)
        .time(timestamp)
        .field("Value",value)        
    )
    #point = (
    #    Point("measurement1")
    #    .tag("tagname1", "tagvalue1")
    #    .field("field1", 0)
    #)
    write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    write_api = write_client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=bucket, org  = "JYLab3F", record=point)
    #query_result = client.query('SELECT * FROM "MachineTemp"."autogen"."temperature"')
    #print(query_result)
    query_result = ""
    write_client.close()

    return query_result

logging.basicConfig(level=logging.ERROR)


def sendinfluxDB(path, matfile) :
    Lat = matfile['Lats_i'][0,0]
    params = matfile['params']
    Lat = (Lat > 0.34).astype('float')
    atomnumber = np.sum(Lat)
    phaseX = params[0,4]
    phaseY = params[0,5]
    photoncount =matfile['photoncounts'][0][0]
    timestamp = datetime.now(timezone.utc) 
    print("Sending ...",)
    SendQueryMachineTemp('AtomNumber', os.path.basename(path), "Lat1", timestamp,atomnumber) 
    SendQueryMachineTemp('LatticePhase', os.path.basename(path), "X", timestamp,phaseX) 
    SendQueryMachineTemp('LatticePhase', os.path.basename(path), "Y", timestamp,phaseY) 
    SendQueryMachineTemp('PhotonCount', os.path.basename(path), "pc1", timestamp,photoncount) 
    print("Done!", atomnumber, phaseX, phaseY, photoncount, timestamp)
    pass
class MyEventHandler(FileSystemEventHandler):
    def __init__(self, observer):
        self.observer = observer
        self.latestFile = ""
        self.count = 0
        self.mailSent = 0
        self.last_modified_time = None  # Store the last modification time

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".mat"):
            if event.src_path != self.latestFile:
                print(f"File created: {event.src_path}")
                self.latestFile = event.src_path
                try:
                    modified_timestamp = os.path.getmtime(self.latestFile)
                    self.last_modified_time = datetime.fromtimestamp(modified_timestamp)
                except FileNotFoundError:
                    print(f"Error: File not found at path: {self.latestFile}")
                except Exception as e:
                    print(f"An error occurred while getting modification time: {e}")
                self.check_and_send_mail(event)

    def check_and_send_mail(self, event):
        file_path = event.src_path
        try:
            time.sleep(1)
            matfile = scipy.io.loadmat(event.src_path)
            data = matfile['photoncounts'][0][0]
            sendinfluxDB(event.src_path, matfile)
            if data < 1000:
                self.count += 1
                print(f"Low photon count detected. Consecutive count: {self.count}")
            else:
                self.count = 0
            if self.count == 3 and self.mailSent == 0:
                self.send_mail(event, "Photon count is below threshold")

        except FileNotFoundError:
            print(f"Error: File not found at path: {file_path}")
        except Exception as e:
            print(f"An error occurred while processing file: {e}")

    def check_for_timeout(self):
        if self.latestFile and self.last_modified_time:
            current_time = datetime.now()
            time_difference = current_time - self.last_modified_time
            if time_difference > timedelta(minutes=5) and self.mailSent == 0:
                self.send_mail(None, "Sequence Timeout") # Pass None for event as we are checking periodically
                self.mailSent = 1
            else:
                print(f"Last .mat file '{self.latestFile}' was modified within the last 5 minutes. Not sending timeout email yet.")
        else:
            print("No .mat file created yet, or last modified time not available.")

    def send_mail(self, event, body):
        subject = "Alert from File System Monitor"
        sender = "hjh79257107@gmail.com"
        recipients = ["gj7107@kaist.ac.kr", "gj7107@naver.com","speedlee1@kaist.ac.kr"]
        password = "czpu mpki fkal dszq"  # It's better to store this securely
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
                smtp_server.login(sender, password)
                smtp_server.sendmail(sender, recipients, msg.as_string())
            print("Message sent!")
            if body == "Photon count is below threshold":
                self.count = 0
                self.mailSent = 1 # Set mailSent to 1 when either email is sent
            elif body == "Sequence Timeout":
                self.mailSent = 1
        except Exception as e:
            print(f"Error sending email: {e}")

if __name__ == '__main__':
    path = r"V:\2025\04\11\Zyla\L12_V04_2"
    observer = Observer()
    event_handler = MyEventHandler(observer)
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(10)  # Check for timeout every 60 seconds (adjust as needed)
            event_handler.check_for_timeout()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()