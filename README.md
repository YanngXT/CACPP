# Dataset

AntiACP2.0_Alternate and AntiACP2.0_Main comes from the server of AntiCP 2.0

The construction of Dataset1, Dataset2, Dataset3, Dataset4, Dataset5 is followed:

| Dataset   | Positive Samples for Train                        | Positive Samples for Test                      | Negative Samples for Train                            | Negative Samples for Test                       |
| --------- | ------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------- |
| Dataset 1 | Randomly selected 80% peptides of all ACPs (777)  | The rest 20% of the peptides of all ACPs (193) | Randomly selected 80% peptides of all non-ACPs (1440) | The rest 20% peptides of all non-ACPs (360)     |
| Dataset 2 | Randomly  selected 60% peptides of all ACPs (582) | Same as Dataset1 (193)                         | Same as Dataset1 (1440)                               | Same as Dataset1 (360)                          |
| Dataset 3 | Randomly selected 40% peptides of all ACPs (387)  | Same as Dataset1 (193)                         | Same as Dataset1 (1440)                               | Same as Dataset1 (360)                          |
| Dataset 4 | Randomly selected 80% peptides of all ACPs (777)  | The rest 20% peptides of all ACPs (193)        | Negative samples in Alternate Dataset for train(776)  | Negative samples in Main Dataset for test (172) |
| Dataset 5 | Same as Dataset 4 (777)                           | Same as Dataset 4 (193)                        | Negative samples in Main Dataset for train (689)      | Same as Dataset 4 (172)                         |
