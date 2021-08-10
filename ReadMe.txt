This folder contains all the data and code used in the paper titled "Seizing Opportunity: Maintenance Optimization in 
Offshore Wind Farms Considering Accessibility, Production and Crew Dispatch.

All the parts of the analysis performed in the paper have been concatenated into a single python notebook: Main_code.ipynb.
To execute the cells in this notebook, all the data and code must be downloaded and saved in the same folder.

- Main_code.ipynb:
	Contains all the code necessery for the analysis of the paper, from data importing, processing and result analysis.

- benchmarks.py:
	Contains the benchmark adaptations of HOST for the comparison of different strategies. Each benchmark is a function
	called within the Main_code.

- {HOST, BESN, PBOS, TBS, CMS}.xlsx:
	Excel files that contain precalculated metrics for 30 weather scenarios used in the analysis of Case Study I in the
	paper.

- {HOST, BESN, PBOS, TBS, CMS}2_10wt_var_price5.json:
	JSON files containing the optimal schedules and metrics obtained by solving for 30 weather scenarios, used in the
	analysis of Case Study II in the paper.

- method_of_bins.csv:
	A csv file containing data used for the binning method and the construction of the power curve.

- da_hrl_lmps_ZONE2014.csv:
	A csv file containing raw hourly electricity price data used in Case Study II, downloaded from PJM data miner 2.

- wind_wave_data.csv:
	A csv file containing pre-processed hourly wind speed and wave height data, downloaded from NYSERDA.



