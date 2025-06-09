import numpy as np
import time
import json
import logging
from kafka import KafkaProducer
from datetime import datetime
import pytz
import uuid

# تنظیم لاگینگ
logging.basicConfig(
    filename='data_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# تنظیمات Kafka
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPIC = 'drilling_data'

# نرخ ارسال
RATE_PER_SECOND = 1

# پارامترهای احتمالی
bit_types = ['PDC', 'Roller Cone', 'Hybrid']
formation_types = ['Sandstone', 'Shale', 'Limestone', 'Carbonate']
shale_reactivity = ['Low', 'Medium', 'High']

def generate_data_record():
    size = 1
    shift = np.random.uniform(-1, 1)
    scale = np.random.uniform(0.8, 1.2)

    depth = np.random.uniform(500, 5000, size)
    viscosity = np.random.normal(30, 10 * scale, size).clip(10, 100)
    mud_weight = np.random.normal(10.5, 1 * scale, size).clip(8.5, 13)

    df = {
        "record_id": str(uuid.uuid4()),
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "Depth_m": float(depth[0]),
        "ROP_mph": float(np.random.normal(20 + shift*2, 8 * scale, size).clip(5, 50)[0]),
        "WOB_kgf": float(np.random.normal(15000 + shift*1000, 5000 * scale, size).clip(5000, 30000)[0]),
        "Torque_Nm": float(np.random.normal(1000 + shift*50, 400 * scale, size).clip(200, 2000)[0]),
        "Pump_Pressure_psi": float(500 + mud_weight * 180 + np.random.normal(0, 300, size)[0]),
        "Mud_FlowRate_LPM": float(10 + (depth / 10) + np.random.normal(0, 100, size)[0]),
        "MWD_Vibration_g": float(np.random.uniform(0.1, 3.0 + shift, size)[0]),
        "Bit_Type": np.random.choice(bit_types),
        "Mud_Weight_ppg": float(mud_weight[0]),
        "Viscosity_cP": float(viscosity[0]),
        "Plastic_Viscosity": float(viscosity[0] * 0.4 + np.random.normal(0, 5)),
        "Yield_Point": float(viscosity[0] * 0.2 + np.random.normal(0, 3)),
        "pH_Level": float(np.random.normal(8.5, 1.2 * scale, size).clip(6.5, 11)[0]),
        "Solid_Content_%": float(np.random.uniform(1, 20, size)[0]),
        "Chloride_Concentration_mgL": float(np.random.normal(50000 + shift*5000, 20000 * scale, size).clip(100, 150000)[0]),
        "Oil_Water_Ratio": float(np.random.uniform(10, 90, size)[0]),
        "Emulsion_Stability": float(np.random.uniform(30, 100, size)[0]),
        "Formation_Type": np.random.choice(formation_types),
        "Pore_Pressure_psi": float(np.random.normal(8000 + shift*500, 2000 * scale, size).clip(3000, 15000)[0]),
        "Fracture_Gradient_ppg": float(np.random.normal(15 + shift*0.2, 1.5 * scale, size).clip(13, 18)[0]),
        "Stress_Tensor_MPa": float(np.random.normal(40 + shift*2, 15 * scale, size).clip(10, 80)[0]),
        "Young_Modulus_GPa": float(np.random.normal(30 + shift*3, 10 * scale, size).clip(5, 70)[0]),
        "Poisson_Ratio": float(np.random.uniform(0.2, 0.35, size)[0]),
        "Brittleness_Index": float(np.random.uniform(0, 1, size)[0]),
        "Shale_Reactiveness": np.random.choice(shale_reactivity),
        "is_operational": np.random.random() < 0.8
    }
    return df

def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    print(f"📡 Sending simulated drilling data to topic '{KAFKA_TOPIC}'...")

    while True:
        try:
            record = generate_data_record()
            producer.send(KAFKA_TOPIC, value=record)
            logging.info(json.dumps(record))
            print(f"✅ Sent record {record['record_id']} at {record['timestamp']}")
            time.sleep(1.0 / RATE_PER_SECOND)
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            print(f"❌ Error: {str(e)}")
            time.sleep(1)

if __name__ == "__main__":
    main()
