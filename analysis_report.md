# Experiment Analysis Report

This report provides a comparative analysis of the conducted experiments.

## Summary of Results

| Experiment                | Model   | Dataset            |   Test Accuracy |   Best Epoch |   Total Epochs |   Training Time (s) |
|:--------------------------|:--------|:-------------------|----------------:|-------------:|---------------:|--------------------:|
| assoc_hrem_baseline_smoke | hrem    | associative_recall |          0.5625 |            1 |              1 |                0.05 |
| assoc_hrem_dnc_smoke      | hrem    | associative_recall |          0.595  |            1 |              1 |                0.07 |
| baseline_mlp_reverse      | mlp     | reverse            |          0.7713 |            3 |              3 |                0.26 |
| copy_hrem_baseline_smoke  | hrem    | copy               |          0.5494 |            1 |              1 |                0.05 |
| copy_hrem_dnc             | hrem    | copy               |          0.7637 |            8 |             15 |              354.27 |
| copy_hrem_dnc_smoke       | hrem    | copy               |          0.4774 |            1 |              1 |                0.06 |
| test_experiment           | N/A     | N/A                |          0.4688 |            1 |              1 |                0.02 |

![Training Loss Comparison](comparison_plot.png)

## Analysis Conclusion

This report summarizes the performance of various models across different tasks. The top-performing model overall was **baseline_mlp_reverse** on the **reverse** task, achieving a test accuracy of **0.7713**.

For the **associative_recall** task, **assoc_hrem_dnc_smoke** was the most effective model. This suggests its architecture may be particularly well-suited for this problem.
For the **copy** task, **copy_hrem_dnc** was the most effective model. This suggests its architecture may be particularly well-suited for this problem.

The accompanying plot of training curves provides further insight into the learning dynamics. Models that achieve a lower final loss more quickly are generally preferable. These initial findings can guide further research, such as more extensive hyperparameter tuning on the most promising models.