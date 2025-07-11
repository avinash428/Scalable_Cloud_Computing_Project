import pandas as pd
from textblob import TextBlob
from collections import Counter
import time
import matplotlib.pyplot as plt
import psutil
import boto3

# Disable GUI backend for matplotlib if needed
import matplotlib
matplotlib.use('Agg')

# ========== S3 Upload Function ==========
s3_client = boto3.client('s3')

def upload_to_s3(local_path, bucket_name, s3_key):
    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"-----> Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"-----> Failed to upload {local_path}: {e}")

# ========== WORD COUNT FUNCTION ==========
def word_count_sequential(df):
    word_counter = Counter()
    for text in df['sentence']:
        words = text.lower().split()
        word_counter.update(words)
    return word_counter

# ========== SENTIMENT ANALYSIS FUNCTION ==========
def sentiment_analysis_sequential(df):
    sentiments = []
    for text in df['sentence']: 
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
        sentiments.append(sentiment)
    return sentiments

# ========== WORKLOAD PROCESSING FUNCTION ==========
def process_workload_sequential(df, workload_size):
    print(f"\n========== Sequential Processing {workload_size} Records ==========")
    df_subset = df.head(workload_size)

    start_time = time.time()

    # CPU & Memory Usage before processing
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().percent

    # WORD COUNT
    word_count_result = word_count_sequential(df_subset)
    print("\nTop 5 Most Common Words:")
    print(word_count_result.most_common(5))

    # SENTIMENT ANALYSIS
    sentiments = sentiment_analysis_sequential(df_subset)
    df_subset['Predicted_Sentiment'] = sentiments

    sentiment_counts = Counter(sentiments)
    print("\nSentiment Counts:")
    print(sentiment_counts)

    # CPU & Memory Usage after processing
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().percent

    total_time = time.time() - start_time
    throughput = workload_size / total_time
    time_per_record = total_time / workload_size

    print(f"\n-----> Processed {workload_size} records in {total_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} tweets/sec")
    print(f"CPU Usage Before: {cpu_before}% | After: {cpu_after}%")
    print(f"Memory Usage Before: {mem_before}% | After: {mem_after}%")
    print(f"Time per Record: {time_per_record:.6f} seconds")

    # Save result CSV
    df_subset.to_csv(f'/home/ec2-user/environment/code/map_reduce_sequential/sequential_output_{workload_size}.csv', index=False)

    return total_time, throughput, cpu_after, mem_after, time_per_record

# ========== MAIN FUNCTION ==========
def main():
    # Load dataset
    df = pd.read_csv('/home/ec2-user/environment/code/archive/train_data.csv')

    workloads = [200000,300000, 500000]
    latencies = []
    throughputs = []
    cpu_usages = []
    memory_usages = []
    time_per_records = []

    for workload in workloads:
        if workload <= len(df):
            total_time, throughput, cpu_usage, mem_usage, time_per_record = process_workload_sequential(df, workload)

            # Append metrics
            latencies.append(total_time)
            throughputs.append(throughput)
            cpu_usages.append(cpu_usage)
            memory_usages.append(mem_usage)
            time_per_records.append(time_per_record)

            # Print summary immediately after each workload
            print(f"\n*** Summary for workload {workload}:")
            print(f"Latency: {total_time:.2f} seconds")
            print(f"Throughput: {throughput:.2f} tweets/sec")
            print(f"CPU Usage After: {cpu_usage:.2f}%")
            print(f"Memory Usage After: {mem_usage:.2f}%")
            print(f"Time per Record: {time_per_record:.6f} seconds\n")

        else:
            print(f"\n*** Skipping workload {workload} — dataset only has {len(df)} records.")

    # Plot Performance Graphs
    plt.figure(figsize=(14, 8))

    # Latency Graph
    plt.figure()
    plt.plot(workloads[:len(latencies)], latencies, marker='o', color='blue')
    plt.title('Latency vs Workload Size (Sequential)')
    plt.xlabel('Number of Records')
    plt.ylabel('Latency (seconds)')
    plt.grid(True)
    plt.savefig('/home/ec2-user/environment/code/map_reduce_sequential/sequential_latency_graph.png')
    upload_to_s3('/home/ec2-user/environment/code/map_reduce_sequential/sequential_latency_graph.png',
                 'scalable-cloud-x23389770', 'mapreduce_sequential_results/sequential_latency_graph.png')
    plt.close()

    # Throughput Graph
    plt.figure()
    plt.plot(workloads[:len(throughputs)], throughputs, marker='o', color='green')
    plt.title('Throughput vs Workload Size (Sequential)')
    plt.xlabel('Number of Records')
    plt.ylabel('Throughput (tweets/sec)')
    plt.grid(True)
    plt.savefig('/home/ec2-user/environment/code/map_reduce_sequential/sequential_throughput_graph.png')
    upload_to_s3('/home/ec2-user/environment/code/map_reduce_sequential/sequential_throughput_graph.png',
                 'scalable-cloud-x23389770', 'mapreduce_sequential_results/sequential_throughput_graph.png')
    plt.close()

    # CPU Usage Graph
    plt.figure()
    plt.plot(workloads[:len(cpu_usages)], cpu_usages, marker='o', color='red')
    plt.title('CPU Usage vs Workload Size (Sequential)')
    plt.xlabel('Number of Records')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.savefig('/home/ec2-user/environment/code/map_reduce_sequential/sequential_cpu_usage_graph.png')
    upload_to_s3('/home/ec2-user/environment/code/map_reduce_sequential/sequential_cpu_usage_graph.png',
                 'scalable-cloud-x23389770', 'mapreduce_sequential_results/sequential_cpu_usage_graph.png')
    plt.close()

    # Memory Usage Graph
    plt.figure()
    plt.plot(workloads[:len(memory_usages)], memory_usages, marker='o', color='purple')
    plt.title('Memory Usage vs Workload Size (Sequential)')
    plt.xlabel('Number of Records')
    plt.ylabel('Memory Usage (%)')
    plt.grid(True)
    plt.savefig('/home/ec2-user/environment/code/map_reduce_sequential/sequential_memory_usage_graph.png')
    upload_to_s3('/home/ec2-user/environment/code/map_reduce_sequential/sequential_memory_usage_graph.png',
                 'scalable-cloud-x23389770', 'mapreduce_sequential_results/sequential_memory_usage_graph.png')
    plt.close()

    # Time per Record Graph
    plt.figure()
    plt.plot(workloads[:len(time_per_records)], time_per_records, marker='o', color='orange')
    plt.title('Time per Record vs Workload Size (Sequential)')
    plt.xlabel('Number of Records')
    plt.ylabel('Time per Record (seconds)')
    plt.grid(True)
    plt.savefig('/home/ec2-user/environment/code/map_reduce_sequential/sequential_time_per_record_graph.png')
    upload_to_s3('/home/ec2-user/environment/code/map_reduce_sequential/sequential_time_per_record_graph.png',
                 'scalable-cloud-x23389770', 'mapreduce_sequential_results/sequential_time_per_record_graph.png')
    plt.close()

if __name__ == '__main__':
    main()
