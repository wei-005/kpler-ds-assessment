# Insights summary / 洞察摘要

## Dataset overview / 数据集概览

- Port calls: 347,250
- Distinct destinations: 2,063
- Distinct vessels: 8,890

- Top-20 busiest ports share: 20.5%; long tail: 79.5%


## Voyage patterns / 航程模式

### Vessel type × distance (top 10)

| vessel_type                      |   trades |   avg_dist_km |
|:---------------------------------|---------:|--------------:|
| Ore/Oil Carrier                  |       92 |      14112.2  |
|                                  |    11746 |       9427.87 |
| Bulk/Oil Carrier (OBO)           |       33 |       9351.21 |
| Bulk Carrier                     |       26 |       8849.71 |
| Crude Oil Tanker                 |    20443 |       8358.14 |
| Bulk/Caustic Soda Carrier (CABU) |       90 |       7955.87 |
| Chemical Tanker                  |     2831 |       7251.56 |
| Crude/Oil Products Tanker        |    93689 |       4717.07 |
| Chemical/Oil Products Tanker     |    63013 |       4716.76 |
| Offshore Tug/Supply Ship         |        4 |       3939.43 |


### Product family × distance (top 10)

| product_family           |   trades |   avg_dist_km |
|:-------------------------|---------:|--------------:|
| clean petroleum products |    90884 |       3692.35 |
| chem/bio                 |    74948 |       6439.15 |
| dirty petroleum products |    35499 |       2924.32 |
| crude oil/condensate     |    35085 |       7687.74 |
| minor bulks              |       35 |       5771.82 |
| lpg                      |       29 |       3243.17 |
| grains/oilseeds          |       24 |       8250.7  |
| coal                     |       15 |       6903.03 |
| olefins                  |       12 |       4475.36 |
| ammonia                  |        3 |       6097.29 |

## Anomalies & special events / 异常与特殊事件

- Trades negative duration: 13
- Trades zero duration: 392
- Inf/NaN speeds: 0
- Speed p95 (m/h): 22,004

- Port calls not linked to trades: 21,217; with cargo>0: 13,878

- STS-like calls: 22,348; canal-related: 433
