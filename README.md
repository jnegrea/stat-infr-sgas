# stat-infr-sgas

## What to run
- One can replicate the experiments in the paper by running the scripts in the `/experiments/scripts` folders. The experiment numbers here do not match the numbers in the paper. See the following table for the correspondance:

| Paper         | Script             |
| ------------- | ------------------ |
| experiment 1  | experiments 1, 1b  |
| experiment 2  | experiments 4, 4b  |
| experiment 3  | experiment 6       |


## Data
- The data for Experiment 2 in the paper is available [here](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/pollock/scale/data_airline_raw.rdata). This is based on the same data from Experiment 3 (below) but has been preprocessed by the Pollock, Fearnhead, and Roberts for experiments in their work: [Quasi-stationary Monte Carlo and the ScaLE algorithm](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12365). The file `data_airline_raw.rdata` should be saved to `/experiments/source-data`.
- The data for Experiment 3 in the paper is available [here](https://community.amstat.org/jointscsg-section/dataexpo/dataexpo2009). It is from the American Statistical Association's Data Expo 2009. The data is compressed, so the individual `YYYY.csv` files should be extracted to `/experiments/source-data/DataExpo2009-extracted`. The bash command `bzip2 -dk *.bz2` will extract all the `YYYY.csv.bz2` files at once.
