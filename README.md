# paperchase
Recommendation system for background reading

# MVP

## Ingestion:

## deal with multiline entries and character breaks in csv:
tr -d '\r' < data/raw/papersum.csv > data/raw/papersum_normalized.csv

Lucene uses Java, so check to see if JVM is installed:
java -version

paperchase uses JDK 17, which can be installed (macos):

`brew install openjdk@17`

Win/linux please refer to openjdk.org

