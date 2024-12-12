# TC-Diffuser
This project includes the code of our deep learning TC forecasting method TC-Diffuser.

## Requirements 
* python 3.8.8
* Pytorch 1.11.0 (GPU)

## Data Preparation
First, we need to download all the data we used in TC-Diffuser.
* TC-Diffuser's [processed datasets](https://pan.baidu.com/s/1b-6QAht46iqJTlo0-VAozA?pwd=TCDI)
* TC-Diffuser's [checkpoint](https://pan.baidu.com/s/1D7YZCfRRlxYF8SWesyv0Kw?pwd=TCDI)

After completing the downloading, move these file to correct file path.
* Move TC-Diffuser's processed datasets to **/TC-Diffuser**, change the **data_dir** in **TC-Diffuser/configs/baseline.yaml** to **TC-Diffuser/processed_data_noise_traj_inten_wind_gph_env**
* Move TC-Diffuser's checkpoint to **/TC-Diffuser/experiment**

## Train
```python
## change the eval_mode in TC-Diffuser/configs/baseline.yaml to False ##
cd TC-Diffuser
python main.py
```

## Test
```python
## change the eval_mode in TC-Diffuser/configs/baseline.yaml to True ##
cd TC-Diffuser
python main.py
```
