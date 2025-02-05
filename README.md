# e-scooter-way

## To ejecute the project
```bash
pip install -r requirements.txt

python multiobjective_main_arguments.py -seed [SEED] -pc [PROB_CROSS] -pm [PROB_FLIP] -POB [POPULTATION] -CPUS [CPUS] -GEN [NUM_GEN]
```
example
```bash
python multiobjective_main_arguments.py -seed 64 -pc 0.9 -pm 0.01 -POB 16 -CPUS 16 -GEN 50
```

Python version 3.10.12 64 bits
## Structure of the project
Main
 * multiobjective_main_arguments.py
 * data-osm
    * Malaga-Subway
        * districts-Malaga-Subway-data-with-nodes.csv
        * map-Malaga-Subway-all--scooter-walking-subway--nearest-path-ONLY-CYCLEWAY.gpkg
        * pair_less_than_3600_new_points.csv

## Execute


The data folder are in https://uma365-my.sharepoint.com/:f:/g/personal/pedroza_uma_es/Ejh3j7c7C79Cqc4KyggCZrIBQZtivYkayUMZMz2yNMr_lg?e=xe16jS
