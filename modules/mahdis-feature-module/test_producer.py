# فایل test_producer.py
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

sample_data = {
    "WOB": 10.5,
    "RPM": 120,
    "Torque": 500,
    "ROP": 8.0,
    "BitArea": 10.0
}

while True:
    producer.send('raw_sensors', sample_data)
    print("Sent:", sample_data)
    time.sleep(2)
