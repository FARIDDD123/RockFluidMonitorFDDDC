from kafka import KafkaConsumer
import json

KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPIC = 'drilling_data'

def main():
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',  # از اول پیام‌ها رو بخون
        enable_auto_commit=True,
        group_id='drilling_data_consumers',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    print(f"🚀 Listening to Kafka topic '{KAFKA_TOPIC}'...")

    for message in consumer:
        data = message.value
        print(f"Received record: {data}")

if __name__ == "__main__":
    main()