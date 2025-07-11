from multiprocessing import Pool, freeze_support, cpu_count
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
import platform
import boto3

s3_client = boto3.client('s3')

def upload_to_s3(local_path, bucket_name, s3_key):
    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"-----> Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"-----> Failed to upload {local_path}: {e}")


def process_sentiment(sentence):
    polarity = TextBlob(sentence).sentiment.polarity
    return 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'

def plot_metric(x, y, title, xlabel, ylabel, color, filename, output_dir):
    plt.figure()
    plt.plot(x, y, color, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path)
    upload_to_s3(full_path, 'scalable-cloud-x23389770', f'kafka_parallel_results/{filename}')
    plt.close()

if __name__ == "__main__":
    freeze_support()

    # Determine platform-specific path
    output_dir = '/home/ec2-user/environment/code/kafka_parallel'
    os.makedirs(output_dir, exist_ok=True)

    workloads = [200000, 300000, 500000]
    batch_size = 1000  # Process 1000 messages at a time

    consumer = KafkaConsumer(
        'tweets',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        group_id='tweet-group'
    )
    print("-----> Kafka consumer connected.")

    pool = Pool(processes=cpu_count())  # use all CPU cores

    all_latencies, all_throughputs = {}, {}
    all_cpu_usages, all_memory_usages, all_time_per_message = {}, {}, {}

    for workload_size in workloads:
        print(f"\n========== Starting workload: {workload_size} records ==========")

        WINDOW_TIME = 30
        window_texts = deque()
        word_counter = Counter()
        sentiment_counter = Counter()

        latencies, throughputs, cpu_usages, memory_usages, time_per_message = [], [], [], [], []
        results = []
        start_time = time.time()
        window_start_time = start_time
        message_count = 0
        message_batch = []

        try:
            for message in consumer:
                message_start = time.time()
                tweet_full = message.value.decode('utf-8')
                try:
                    sentence, real_sentiment = tweet_full.split(":::")
                except ValueError:
                    continue

                message_count += 1
                message_batch.append((sentence, real_sentiment))

                # Word counting locally (fast, no need for multiprocessing)
                window_texts.append((time.time(), sentence))
                word_counter.update(sentence.lower().split())

                while window_texts and (time.time() - window_texts[0][0]) > WINDOW_TIME:
                    old_msg_time, old_msg = window_texts.popleft()
                    word_counter.subtract(old_msg.lower().split())

                # When batch is ready, process sentiments in parallel
                if len(message_batch) >= batch_size or message_count == workload_size:
                    sentences = [msg[0] for msg in message_batch]
                    sentiments = pool.map(process_sentiment, sentences)

                    for idx, sentiment in enumerate(sentiments):
                        sentiment_counter[sentiment] += 1
                        sentence, real_sentiment = message_batch[idx]
                        results.append([sentence, real_sentiment, sentiment])

                    batch_time = time.time() - message_start
                    latencies.append(batch_time)
                    time_per_message.append(batch_time / len(message_batch))

                    if int(time.time() - window_start_time) % 5 == 0:
                        cpu_usages.append(psutil.cpu_percent(interval=0.2))
                        memory_usages.append(psutil.virtual_memory().percent)

                    if int(time.time() - start_time) % 10 == 0:
                        throughput = len(window_texts) / WINDOW_TIME
                        throughputs.append(throughput)

                    message_batch.clear()

                if message_count >= workload_size:
                    print(f"*** completed {workload_size} messages.")
                    break

        except KeyboardInterrupt:
            print("\n**** Interrupted by user.")

        workload_time = time.time() - start_time

        result_path = os.path.join(output_dir, f'stream_output_parallel_{workload_size}.csv')
        pd.DataFrame(results, columns=['Sentence', 'Real_Sentiment', 'Predicted_Sentiment']).to_csv(result_path, index=False)
        print(f"----> Results saved to {result_path}")

        all_latencies[workload_size] = latencies
        all_throughputs[workload_size] = throughputs
        all_cpu_usages[workload_size] = cpu_usages
        all_memory_usages[workload_size] = memory_usages
        all_time_per_message[workload_size] = time_per_message

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0
        avg_time_per_msg = sum(time_per_message) / len(time_per_message) if time_per_message else 0

        print(f"\n*** Stats for workload {workload_size}:")
        print("Top 5 words:", word_counter.most_common(5))
        print("Sentiment distribution:", sentiment_counter)
        print(f"Average Latency: {avg_latency:.4f} s")
        print(f"Average Throughput: {avg_throughput:.2f} tweets/sec")
        print(f"Average CPU Usage: {avg_cpu:.2f}%")
        print(f"Average Memory Usage: {avg_memory:.2f}%")
        print(f"Average Time/Message: {avg_time_per_msg:.6f} s")
        print(f"Total Processing Time: {workload_time:.2f} s\n")

    pool.close()
    pool.join()

    avg_cpu_usages = {w: sum(u) / len(u) if u else 0 for w, u in all_cpu_usages.items()}
    avg_memory_usages = {w: sum(m) / len(m) if m else 0 for w, m in all_memory_usages.items()}
    avg_latencies = {w: sum(l) / len(l) if l else 0 for w, l in all_latencies.items()}
    avg_throughputs = {w: sum(t) / len(t) if t else 0 for w, t in all_throughputs.items()}
    avg_time_per_message = {w: sum(t) / len(t) if t else 0 for w, t in all_time_per_message.items()}

    # Final combined plots
    plot_metric(list(avg_cpu_usages.keys()), list(avg_cpu_usages.values()),
                "CPU Usage vs Workload Size", "Records", "CPU (%)", 'ro-', "cpu_vs_workload.png", output_dir)

    plot_metric(list(avg_latencies.keys()), list(avg_latencies.values()),
                "Latency vs Workload Size", "Records", "Latency (s)", 'bo-', "latency_vs_workload.png", output_dir)

    plot_metric(list(avg_throughputs.keys()), list(avg_throughputs.values()),
                "Throughput vs Workload Size", "Records", "Tweets/sec", 'go-', "throughput_vs_workload.png", output_dir)

    plot_metric(list(avg_memory_usages.keys()), list(avg_memory_usages.values()),
                "Memory Usage vs Workload Size", "Records", "Memory (%)", 'mo-', "memory_vs_workload.png", output_dir)

    plot_metric(list(avg_time_per_message.keys()), list(avg_time_per_message.values()),
                "Time/Message vs Workload Size", "Records", "Time/msg (s)", 'co-', "time_per_message_vs_workload.png", output_dir)

    print("\n-----> workloads processed, results saved, and combined performance graphs generated.")
    consumer.close()
