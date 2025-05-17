# paperchase
Recommendation system for background reading

# MVP

## Ingestion:

Dataset has two significant out-of-the-box processing issues:

- Fields with multiple new lines getting broken up into separate fields, misaligning the dataset
    To solve, install csvkit. In macos: `brew install csvkit`
    Run `csvclean --lenient -a paperSum.csv`

- Some abstract fields are empty, which pandas processes as NaN. build_index.py script accounts for this by safely parsing and returning an empty string for the field. 

Lucene uses Java, so check to see if JVM is installed:
java -version

paperchase uses JDK 17, which can be installed (macos):

`brew install openjdk@17`

Win/linux please refer to openjdk.org

References (with paper has dataset download):

- Liu, J., Vats, A., & He, Z. (2025). CS-PaperSum: A Large-Scale Dataset of AI-Generated Summaries for Scientific Papers. arXiv:2502.20582. [![arXiv](https://img.shields.io/badge/arXiv-2502.20582-b31b1b.svg)](https://arxiv.org/abs/2502.20582)