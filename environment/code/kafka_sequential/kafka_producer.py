from kafka import KafkaProducer
import pandas as pd
import time

# Load dataset
df = pd.read_csv('/home/ec2-user/environment/code/archive/train_data.csv')

# Connect to Kafka
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    batch_size=16384,       # increase to 16KB
    linger_ms=10,           # wait up to 10ms to batch messages
    buffer_memory=33554432  # 32MB buffer memory
)
print(f"----->Kafka producer connected. Sending tweets...")

# Send messages to 'tweets' topic
for index, row in df.iterrows():
    message = f"{row['sentence']}:::{row['sentiment']}"
    producer.send('tweets', value=message.encode('utf-8'))


print("----->All tweets sent.")
producer.close()
