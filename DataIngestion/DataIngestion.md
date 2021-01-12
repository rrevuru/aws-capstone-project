# Data Ingestion & Transformation
## About Dataset
This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.
The dataset is split amongst the following 4 csv files
 * links.csv, 
*  movies.csv, 
*  ratings.csv 
*  tags.csv

## Data Ingestion Phase 
I have downloaded the dataset onto a EC2 server 
### Kinesis Agent
Installed a kineses agent, and configured the agent to load files from the source. Since this is not a streaming application, we had to use the "START_OF_THE_FILE" to read the whole file
![image](kinesisagent.png)

### Kinesis Firehose Streams
Configured four firehose streams, corresponding to each files within the directories.

![image](kinesisfirehose.png)


#### Destination
I  had to create 4 S3 buckets inorder to transport the files from the source location to the destination S3 bucket

![image](S3buckets.png)