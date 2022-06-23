# HOST: Opportunistic Maintenance Optimization in Offshore Wind Farms

This folder contains all the data and code used in:

Papadopoulos, P., Coit, D.W. and Ezzat, A.A., 2022. Seizing Opportunity: Maintenance Optimization in Offshore Wind Farms Considering Accessibility, Production, and Crew Dispatch. *IEEE Transactions on Sustainable Energy, 13(1)*, pp.111-121.

**Please use the reference above if you use parts of the code or data**

All the parts of the analysis performed in the paper have been concatenated into a single python notebook: <code>Main_code.ipynb</code>
To execute the cells in this notebook, all the data and code must be downloaded and saved in the same folder.

- <code>Main_code.ipynb</code>:
	Contains all the code necessery for the analysis of the paper, from data importing, processing and result analysis.

- <code>benchmarks.py</code>:
	Contains the benchmark adaptations of HOST for the comparison of different strategies. Each benchmark is a function
	called within the Main_code.

- <code>{HOST, BESN, PBOS, TBS, CMS}.xlsx</code>:
	Excel files that contain precalculated metrics for 30 weather scenarios used in the analysis of Case Study I in the
	paper.

- <code>{HOST, BESN, PBOS, TBS, CMS}2_10wt_var_price5.json</code>:
	JSON files containing the optimal schedules and metrics obtained by solving for 30 weather scenarios, used in the
	analysis of Case Study II in the paper.

- <code>method_of_bins.csv</code>:
	A csv file containing data used for the binning method and the construction of the power curve.

- <code>da_hrl_lmps_ZONE2014.csv</code>:
	A csv file containing raw hourly electricity price data used in Case Study II, downloaded from PJM data miner 2.

- <code>wind_wave_data.csv</code>:
	A csv file containing pre-processed hourly wind speed and wave height data, downloaded from NYSERDA.
