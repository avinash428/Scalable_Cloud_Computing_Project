from kafka import KafkaConsumer
from textblob import TextBlob
from collections import Counter, deque
import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')

import boto3

s3_client = boto3.client('s3')

def upload_to_s3(local_path, bucket_name, s3_key):
    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"-----> Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"-----> Failed to upload {local_path}: {e}")

# Create result directory
os.makedirs('/home/ec2-user/environment/code/kafka_sequential', exist_ok=True)

# Workloads to test
workloads = [200000, 300000, 500000]

# Connect to Kafka
consumer = KafkaConsumer(
    'tweets',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='tweet-group'
)

print("-----> Kafka consumer connected. Listening for tweets...")

# Combined results
avg_latencies = []
avg_throughputs = []
avg_cpu_usages = []
avg_memory_usages = []
total_times = []
time_per_records = []

for workload_size in workloads:
    print(f"\n*** Starting workload: {workload_size} records")

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
                print(f"\n-----> Completed {workload_size} messages.")
                break

    except KeyboardInterrupt:
        print("\n-----> Interrupted by user.")

    workload_end_time = time.time()
    total_time_taken = workload_end_time - workload_start_time
    time_per_record = total_time_taken / workload_size

    avg_latencies.append(sum(latencies) / len(latencies) if latencies else 0)
    avg_throughputs.append(sum(throughputs) / len(throughputs) if throughputs else 0)
    avg_cpu_usages.append(sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0)
    avg_memory_usages.append(sum(memory_usages) / len(memory_usages) if memory_usages else 0)
    total_times.append(total_time_taken)
    time_per_records.append(time_per_record)

    print(f"-----> Stats for {workload_size} messages:")
    print(f"Average Latency: {avg_latencies[-1]:.4f} s")
    print(f"Average Throughput: {avg_throughputs[-1]:.2f} tweets/sec")
    print(f"CPU Usage: {avg_cpu_usages[-1]:.2f}%")
    print(f"Memory Usage: {avg_memory_usages[-1]:.2f}%")
    print(f"Total Time: {total_time_taken:.2f} s")
    print(f"Time per Record: {time_per_record:.6f} s")

# ========== Combined Performance Graphs ==========

def plot_and_upload(x, y, title, xlabel, ylabel, color, filename):
    plt.figure()
    plt.plot(x, y, color, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    full_path = f"/home/ec2-user/environment/code/kafka_sequential/{filename}"
    plt.savefig(full_path)
    upload_to_s3(full_path, 'scalable-cloud-x23389770', f'kafka_sequential_results/{filename}')
    plt.close()

plot_and_upload(workloads, avg_latencies, "Latency vs Workload Size",
                "Number of Records", "Latency (seconds)", 'bo-', "latency.png")

plot_and_upload(workloads, avg_throughputs, "Throughput vs Workload Size",
                "Number of Records", "Throughput (tweets/sec)", 'go-', "throughput.png")

plot_and_upload(workloads, avg_cpu_usages, "CPU Usage vs Workload Size",
                "Number of Records", "CPU Usage (%)", 'ro-', "cpu_usage.png")

plot_and_upload(workloads, avg_memory_usages, "Memory Usage vs Workload Size",
                "Number of Records", "Memory Usage (%)", 'mo-', "memory_usage.png")

plot_and_upload(workloads, total_times, "Total Time vs Workload Size",
                "Number of Records", "Total Time (seconds)", 'ko-', "total_time.png")

plot_and_upload(workloads, time_per_records, "Time per Record vs Workload Size",
                "Number of Records", "Time per Record (seconds)", 'co-', "time_per_record.png")

print("\n-----> All workloads processed and combined performance graphs saved.")
consumer.close()
