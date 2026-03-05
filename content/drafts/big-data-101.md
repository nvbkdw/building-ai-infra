---
title: "Big Data 101"
date: 2026-03-05
tags: ["big data"]
author: "Ryan H."
description: "This blog post covers the big data 101."
summary: "This blog post covers the big data 101."
cover:
    image: "big-data-101.png"
    alt: "Big Data 101"
    relative: true
---

# Brief History of Big Data

### 1. The Data Locality Era (Early 2000s)

* **The Problem:** In the late 90s and early 2000s, companies like Google were dealing with explosive web growth. Traditionally, data was kept in storage and moved over a network to a server for processing (a decoupled architecture). However, network speeds were incredibly slow, making the transfer a massive bottleneck.
* **The Solution (Hadoop & MapReduce):** Google designed a system to chop files up and distribute them across a cluster. Instead of moving data to the compute, they moved the *compute to the data*. This is called **data locality**. This research birthed the open-source **Hadoop** project and **MapReduce**, a Java-based programming model that processed data right where it lived on the physical disks.
* **Fault Tolerance:** Because Hadoop ran on cheap commodity servers that failed frequently, it wrote intermediate results to disk at every stage. If a server crashed, it could easily pick up where it left off.

### 2. The SQL Abstraction Era

* **The Problem:** Facebook adopted Hadoop to process hundreds of petabytes of data, but most of their analysts didn't know how to write complex Java MapReduce programs.
* **The Solution (Hive):** Facebook created **Hive**, an abstraction layer that allowed users to write simple SQL queries. Hive automatically translated that SQL into the complex, low-level Java MapReduce pipelines under the hood.
* **The Bottleneck:** While Hive made data accessible, it still relied on Hadoop's disk-heavy fault tolerance. Constantly reading and writing to physical disks made Hive queries very slow.

### 3. The In-Memory Era (2010s)

MapReduce was revolutionary because it allowed distributed processing, but it had a fatal flaw: disk I/O. To ensure that a job didn't completely fail if one cheap server crashed, MapReduce wrote the intermediate results of every single step to physical hard drives.

* **Spark:** Spark was born out of UC Berkeley around 2009 specifically to fix this speed problem. Spark introduced in-memory processing. Instead of writing intermediate data back to a hard drive, Spark caches that data in the RAM of its worker nodes, making it up to 100x faster than Hadoop for certain workloads. Spark maintained **fault tolerance** by tracking the "lineage" of transformations rather than saving intermediate steps to disk.

* **Presto (later Trino):** Facebook needed fast, interactive querying on massive datasets, and Hive's disk-writing made that impossible. Furthermore, network speeds had fundamentally changed—jumping from millions of bits per second in 2001 to billions of bits per second in 2012. Because networks were fast again, architectures could return to being decoupled. Presto/Trino is designed to move data over the network to compute engines for incredibly fast, ad-hoc queries by doing all processing **in-memory**. To achieve this speed, it sacrificed mid-query fault tolerance; if a node failed, the query just aborted and had to be rerun.

#### Spark vs. Trino Today: The Heavy Lifter vs. The Quick Reader

* Trino/Presto is the fast, federated SQL engine. It's built for business analysts who want to write a quick query to join a database and a massive cloud table to see a result in 5 seconds. However, if you try to use Trino to transform a petabyte of raw data, it will likely suffer an "Out of Memory" (OOM) crash.

* Spark is the heavy-duty ETL (Extract, Transform, Load) engine. Data engineers use Spark to run massive, complex, overnight data pipelines. It might take longer to spin up than a quick Trino query, but because of its architecture, it will chew through petabytes of data reliably without crashing.


### 4. The Modern Federated & Cloud Era

* **Cloud Storage:** Today's platforms use a shared-data architecture. Storage and compute are completely decoupled and scale independently. Data lives in cheap cloud object storage (like Azure Blob) and compute clusters are spun up to process it over high-speed networks.

* **Federated Querying (Trino):** As data exploded across different formats (relational databases, NoSQL like MongoDB, streams like Kafka), moving all of it into one central warehouse via ETL jobs became a bottleneck. Trino evolved into a "universal translator," allowing users to write one SQL query that reaches across these disparate, decoupled systems without having to move the data first.

---

The overarching theme is that as network speeds caught up to processing demands, Big Data moved from clunky, disk-heavy, localized processing to fast, in-memory, decoupled, and federated networks.

Would you like me to dive deeper into how Trino's "push down" optimization actually works across those different database types?


# DataLake: Separated Storage Layer
DataLake is a new storage layer that separates the data from the computation. It is a new storage layer that is designed to store data in a way that is efficient for both read and write operations. It is a new storage layer that is designed to store data in a way that is efficient for both read and write operations. It is a new storage layer that is designed to store data in a way that is efficient for both read and write operations.

Datalake Table Formats (Delta Lake, Iceberg, Hudi)

TBD: explain separated compute and storage layer architecture

## Iceberg
TBD: explain Iceberg architecture

### Small File Problem
The struggle you are experiencing is one of the most famous and universally frustrating issues in distributed data processing. It is affectionately (and sometimes bitterly) known in the data engineering world as the **"Small File Problem."**

Apache Spark is designed to process massive datasets by breaking them down into large, manageable chunks. When you feed it millions of tiny files instead of a few large ones, it breaks Spark's core architecture in a few fundamental ways.

Here is exactly why those small files cause immense memory pressure and slow down your processing to a crawl.

---

### 1. Driver Memory Overload (The "Memory Pressure")

In Spark, the **Driver** node acts as the brain of your application. Before the worker nodes (Executors) can process any data, the Driver has to figure out exactly where all the data lives.

* **Metadata Explosion:** For every single file you read, the Driver must store its metadata (file path, size, block locations, partition information) in its own JVM heap memory.
* **Out of Memory (OOM):** If you try to load 10 large files, the Driver tracks 10 objects. If you try to load 1,000,000 small files (even if the total data size is the same), the Driver has to keep track of a million objects. This fills up the Driver's memory rapidly, leading to constant Garbage Collection (which pauses your job) or a complete crash (`java.lang.OutOfMemoryError`).

### 2. Task Scheduling Overhead (The "Slowness")

By default, Spark creates **one partition for every file** it reads (unless the files are exceptionally large, in which case it splits them, but it never naturally merges small ones upon reading unless specifically instructed).

* **1 File = 1 Partition = 1 Task:** If you have 100,000 files, Spark will try to schedule and launch 100,000 tasks.
* **Setup takes longer than execution:** Launching a task, serializing the data, sending it over the network to an Executor, and spinning up the compute resources takes a few milliseconds per task. If your file is only 50KB, it might take 1 millisecond to actually process the data, but 50 milliseconds just to *set up* the task. You end up spending 98% of your computing time on administrative overhead rather than actual data processing.

### 3. Storage System Bottlenecks

The problem isn't just Spark; it's the storage system holding your files.

* **Cloud Storage (S3, GCS, Azure):** Cloud storage systems are object stores accessed via REST APIs. Opening connections and executing `GET` or `LIST` requests for millions of tiny objects will cause you to hit API rate limits, resulting in throttling, retries, and massive delays.
* **HDFS (Hadoop Distributed File System):** If you are on-premise, the HDFS NameNode stores file metadata in its RAM. Millions of small files will max out the NameNode's RAM, potentially crashing the entire Hadoop cluster for everyone else.

### 4. Physical Disk I/O and Poor Compression

* **Seek Time vs. Read Time:** Reading data requires the storage system to "seek" the beginning of the file. With thousands of small files, the system spends most of its time jumping around the disk (or making new network requests) to find the next file, rather than doing continuous, high-speed reading.
* **Useless Compression:** Compression algorithms (like Snappy, Gzip, or Parquet's internal compression) rely on finding patterns in data. A 10KB file doesn't have enough data to build a good compression dictionary, meaning your files take up more space on disk than they would if they were merged.

---

### The Goal: The "Goldilocks" File Size

Spark operates best when files are between **128 MB and 1 GB** in size. This perfectly aligns with standard block sizes in distributed storage and gives Spark Tasks enough "meat" to process efficiently without overwhelming memory.

Luckily, with the Modern Solution: Table Formats (Delta Lake, Iceberg, Hudi), you can fix this problem easily.
If you are building a modern data lake, you should absolutely be using a table format like Delta Lake, Apache Iceberg, or Apache Hudi. These formats have built-in, highly optimized commands specifically designed to fix this exact problem.

Compaction / Bin-Packing: In Delta Lake, for example, you can run a simple SQL command: OPTIMIZE table_name;. Delta Lake will look at all your tiny files, spin up a fast, highly-optimized job to read them, merge them into perfect 1GB Parquet files, and update the transaction log—all without you having to write any complex Spark logic.

Auto-Compaction: Many of these formats can be configured to automatically run small background compaction jobs every time you write data, meaning you never even have to think about small files again.
