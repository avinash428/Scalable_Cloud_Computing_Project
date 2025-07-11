from kafka import KafkaConsumer
from textblob import TextBlob
from collections import Counter, deque
import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import os



import boto3

s3_client = boto3.client('s3')

def upload_to_s3(local_path, bucket_name, s3_key):
    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"----->Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"----->Failed to upload {local_path}: {e}")



# Create result directories if not exist
os.makedirs('stream_results_combined', exist_ok=True)

# Workloads to test
workloads = [500000, 1000000, 1500000]

# Connect to Kafka
consumer = KafkaConsumer(
    'tweets',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='tweet-group'
)

print("----->Kafka consumer connected. Listening for tweets...")

# Combined results
avg_latencies = []
avg_throughputs = []
avg_cpu_usages = []
avg_memory_usages = []
total_times = []

for workload_size in workloads:
    print(f"\n⚙️ Starting workload: {workload_size} records")

    # Initialize metrics
    WINDOW_TIME = 30
    start_time = time.time()
    window_texts = deque()

    word_counter = Counter()
    sentiment_counter = Counter()

    latencies, throughputs, cpu_usages, memory_usages = [], [], [], []
    message_count = 0

    workload_start_time = time.time()

    try:
        for message in consumer:
            message_start = time.time()
            tweet_full = message.value.decode('utf-8')
            message_count += 1

            try:
                sentence, real_sentiment = tweet_full.split(":::")
            except ValueError:
                continue

            window_texts.append((time.time(), sentence))
            while window_texts and (time.time() - window_texts[0][0]) > WINDOW_TIME:
                old_msg_time, old_msg = window_texts.popleft()
                word_counter.subtract(old_msg.lower().split())

            word_counter.update(sentence.lower().split())

            polarity = TextBlob(sentence).sentiment.polarity
            predicted_sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
            sentiment_counter[predicted_sentiment] += 1

            latencies.append(time.time() - message_start)

            if int(time.time() - workload_start_time) % 5 == 0:
                cpu_usages.append(psutil.cpu_percent(interval=0.5))
                memory_usages.append(psutil.virtual_memory().percent)

            if int(time.time() - workload_start_time) % 10 == 0:
                throughput = len(window_texts) / WINDOW_TIME
                throughputs.append(throughput)

            if message_count >= workload_size:
                print(f"\n----->Completed {workload_size} messages.")
                break

    except KeyboardInterrupt:
        print("\n----->Interrupted by user.")

    workload_end_time = time.time()
    total_time_taken = workload_end_time - workload_start_time

    avg_latencies.append(sum(latencies) / len(latencies) if latencies else 0)
    avg_throughputs.append(sum(throughputs) / len(throughputs) if throughputs else 0)
    avg_cpu_usages.append(sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0)
    avg_memory_usages.append(sum(memory_usages) / len(memory_usages) if memory_usages else 0)
    total_times.append(total_time_taken)

    print(f"----->Stats for {workload_size} messages: Latency={avg_latencies[-1]:.4f}s, "
          f"Throughput={avg_throughputs[-1]:.2f}, CPU={avg_cpu_usages[-1]:.2f}%, "
          f"Memory={avg_memory_usages[-1]:.2f}%, Total Time={total_time_taken:.2f}s")

# ========== Combined Performance Graphs ==========

# Latency vs Workload
plt.figure()
plt.plot(workloads, avg_latencies, marker='o', color='blue')
plt.title('Average Latency vs Workload Size')
plt.xlabel('Number of Records')
plt.ylabel('Latency (seconds)')
plt.grid(True)
plt.savefig('/home/ec2-user/environment/code/kafka_sequential/combined_latency.png')
path = '/home/ec2-user/environment/code/kafka_sequential/combined_latency.png'
upload_to_s3(path, 'scalable-cloud-x23389770', 'kafka_sequential_results/combined_latency.png')
plt.close()

# Throughput vs Workload
plt.figure()
plt.plot(workloads, avg_throughputs, marker='o', color='green')
plt.title('Average Throughput vs Workload Size')
plt.xlabel('Number of Records')
plt.ylabel('Throughput (tweets/sec)')
plt.grid(True)
plt.savefig('/home/ec2-user/environment/code/kafka_sequential/combined_throughput.png')
path = '/home/ec2-user/environment/code/kafka_sequential/combined_throughput.png'
upload_to_s3(path, 'scalable-cloud-x23389770', 'kafka_sequential_results/combined_throughput.png')
plt.close()

# CPU Usage vs Workload
plt.figure()
plt.plot(workloads, avg_cpu_usages, marker='o', color='red')
plt.title('Average CPU Usage vs Workload Size')
plt.xlabel('Number of Records')
plt.ylabel('CPU Usage (%)')
plt.grid(True)
plt.savefig('/home/ec2-user/environment/code/kafka_sequential/combined_cpu_usage.png')
path = '/home/ec2-user/environment/code/kafka_sequential/combined_cpu_usage.png'
upload_to_s3(path, 'scalable-cloud-x23389770', 'kafka_sequential_results/combined_cpu_usage.png')
plt.close()

# Memory Usage vs Workload
plt.figure()
plt.plot(workloads, avg_memory_usages, marker='o', color='purple')
plt.title('Average Memory Usage vs Workload Size')
plt.xlabel('Number of Records')
plt.ylabel('Memory Usage (%)')
plt.grid(True)
plt.savefig('/home/ec2-user/environment/code/kafka_sequential/combined_memory_usage.png')
path = '/home/ec2-user/environment/code/kafka_sequential/combined_memory_usage.png'
upload_to_s3(path, 'scalable-cloud-x23389770', 'kafka_sequential_results/combined_memory_usage.png')
plt.close()

# Total Time vs Workload
plt.figure()
plt.plot(workloads, total_times, marker='o', color='orange')
plt.title('Total Processing Time vs Workload Size')
plt.xlabel('Number of Records')
plt.ylabel('Total Time (seconds)')
plt.grid(True)
plt.savefig('/home/ec2-user/environment/code/kafka_sequential/combined_total_time.png')
path = '/home/ec2-user/environment/code/kafka_sequential/combined_total_time.png'
upload_to_s3(path, 'scalable-cloud-x23389770', 'kafka_sequential_results/combined_total_time.png')
plt.close()

print("\n*** All workloads processed and combined performance graphs saved.")
consumer.close()
