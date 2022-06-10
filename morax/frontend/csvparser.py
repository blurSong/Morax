# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0318
# Modify and parse .csv file

import os
import re
import sys
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import subprocess as SP
from torch._C import CONV_BN_FUSION
from morax.model.layer import morax_linearlayer_type_dicts as MXLTD


def add_pooling_to_csv(_modelpath, _modelname, _isbn=False):
    # add pooling layer to csv and reconstruct csv index
    # mora.csv -> morax.csv
    if _isbn is True:
        moracsvname = _modelname + "_norm.csv"
        moraxcsvname = _modelname + "_norm_pl.csv"
    else:
        moracsvname = _modelname + ".csv"
        moraxcsvname = _modelname + "_pl.csv"
    mora_csv_path = os.path.abspath(
        os.path.join(_modelpath, _modelname + "/" + moracsvname)
    )
    morax_csv_path = os.path.abspath(
        os.path.join(_modelpath, _modelname + "/" + moraxcsvname)
    )
    model_df = pd.read_csv(mora_csv_path)
    # 1. refill IDX and APD
    # 2. refill RP and add pooling layer
    for idx, layer in model_df.iterrows():
        if layer["IDX"] != -1:
            for tmpidx in range(layer["IDX"] + 1, 0, 1):  # -IDX + 1 ~ -1
                if model_df.at[idx + tmpidx, "RP"] >= 2:
                    model_df.at[idx, "IDX"] -= 1
        if apdidx2_is_index2(MXLTD[layer["TYP"]], layer["APD"]):
            for tmpidx in range(layer["APD"] + 1, 0, 1):
                if model_df.at[idx + tmpidx, "RP"] >= 2:
                    model_df.at[idx, "APD"] -= 1
    for idx, layer in model_df.iterrows():
        if layer["RP"] >= 2:
            poolingrow = {
                "IC": layer["OC"],
                "OC": layer["OC"],
                "FS": layer["FS"],
                "KS": layer["RP"],
                "STR": layer["RP"],
                "TYP": -1,  # MXNTD
                "RP": 0,
                "IDX": -1,
                "APD": 0,
            }
            poolingrow_df = pd.DataFrame(poolingrow, index=[idx + 1])
            model_df = insert_a_row(model_df, poolingrow_df, idx + 1)
            model_df.at[idx, "RP"] = 1
        if MXLTD[layer["TYP"]] == "Linear" and layer["APD"] > 1:
            prelayer = model_df.iloc[idx - 1, :]
            poolingrow = {
                "IC": prelayer["OC"],
                "OC": prelayer["OC"],
                "FS": prelayer["FS"],
                "KS": layer["APD"],
                "STR": layer["APD"],
                "TYP": -1,
                "RP": 0,
                "IDX": -1,
                "APD": 0,
            }
            poolingrow_df = pd.DataFrame(poolingrow, index=[idx + 1])
            model_df = insert_a_row(model_df, poolingrow_df, idx)
            model_df.at[idx, "APD"] = 1
    model_df.to_csv(morax_csv_path, index=False)
    print("[Morax][Front] reconstruct mora.csv -> morax.csv")
    return


def remove_bn_to_csv(_modelpath, _model):
    # remove bn layer and reconstruct csv for 2 simulators
    model_csv_path = os.path.abspath(
        os.path.join(_modelpath, _model + "/" + _model + "_norm.csv")
    )
    model_csv_path_nobn = os.path.abspath(
        os.path.join(_modelpath, _model + "/" + _model + ".csv")
    )
    model_df = pd.read_csv(model_csv_path)
    # 1. refill ReLU pooling to previous layer
    # 2. refill index to next layer
    for idx, layer in model_df.iterrows():
        if MXLTD[layer["TYP"]] == "Batchnorm":
            assert layer["IDX"] == -1, "[Morax][Front] Batchnorm idx is not -1."
            assert (
                model_df.at[idx + layer["IDX"], "RP"] == 0
            ), "[Mora][Front] ConvBNReLU, conv layer {0} RP is not 0.".format(
                idx + layer["IDX"]
            )
            model_df.at[idx + layer["IDX"], "RP"] = layer["RP"]
            model_df.at[idx, "RP"] = 0
    for idx, layer in model_df.iterrows():
        if layer["IDX"] != -1:
            for tmpidx in range(layer["IDX"] + 1, 0, 1):
                if MXLTD[model_df.at[idx + tmpidx, "TYP"]] == "Batchnorm":
                    model_df.at[idx, "IDX"] += 1
                # old problem from noob MNSIM
                # if model_df.at[idx + tmpidx, 'RP'] > 0:
                #    model_df.at[idx, 'IDX'] -= 1 if model_df.at[idx + tmpidx, 'RP'] == 1 else 2
        if apdidx2_is_index2(MXLTD[layer["TYP"]], layer["APD"]):
            for tmpidx in range(layer["APD"] + 1, 0, 1):
                if MXLTD[model_df.at[idx + tmpidx, "TYP"]] == "Batchnorm":
                    model_df.at[idx, "APD"] += 1
                # if model_df.at[idx + tmpidx, 'RP'] > 0:
                #    layer['APD'] -= 1 if model_df.at[idx + tmpidx, 'RP'] == 1 else 2
    model_df = model_df.drop(model_df[model_df["TYP"] == 4].index)
    model_df.to_csv(model_csv_path_nobn, index=False)
    print("[Morax][Front] remove Batchnorm model_norm.csv -> model.csv.")
    return


def remove_ln_to_csv(_modelpath, _model):
    # remove bn layer and reconstruct csv for 2 simulators
    model_csv_path = os.path.abspath(
        os.path.join(_modelpath, _model + "/" + _model + "_norm.csv")
    )
    model_csv_path_nobn = os.path.abspath(
        os.path.join(_modelpath, _model + "/" + _model + ".csv")
    )
    model_df = pd.read_csv(model_csv_path)
    # 1. refill ReLU pooling to previous layer
    # 2. refill index to next layer
    for idx, layer in model_df.iterrows():
        if MXLTD[layer["TYP"]] == "Layernorm":
            assert layer["IDX1"] == -1, "[Morax][Front] Layernorm idx is not -1."
            model_df.at[idx + layer["IDX"], "ACT"] = layer["ACT"]
            model_df.at[idx, "ACT"] = 0
    for idx, layer in model_df.iterrows():
        if layer["IDX1"] != -1:
            for tmpidx in range(layer["IDX1"] + 1, 0, 1):
                if MXLTD[model_df.at[idx + tmpidx, "TYP"]] == "Layernorm":
                    model_df.at[idx, "IDX1"] += 1
                # old problem from noob MNSIM
                # if model_df.at[idx + tmpidx, 'RP'] > 0:
                #    model_df.at[idx, 'IDX'] -= 1 if model_df.at[idx + tmpidx, 'RP'] == 1 else 2
        if apdidx2_is_index2(MXLTD[layer["TYP"]], layer["IDX2"]):
            for tmpidx in range(layer["IDX2"] + 1, 0, 1):
                if MXLTD[model_df.at[idx + tmpidx, "TYP"]] == "Layernorm":
                    model_df.at[idx, "IDX2"] += 1
                # if model_df.at[idx + tmpidx, 'RP'] > 0:
                #    layer['APD'] -= 1 if model_df.at[idx + tmpidx, 'RP'] == 1 else 2
    model_df = model_df.drop(model_df[model_df["TYP"] == 13].index)
    model_df.to_csv(model_csv_path_nobn, index=False)
    print("[Morax][Front] remove Layernorm model_norm.csv -> model.csv.")
    return


def apdidx2_is_index2(type, apd):
    return (
        type == "CONV"
        or type == "Residual"
        or type == "VDP"
        or type == "VADD"
        or type == "VMUL"
        or type == "VMM"
        or type == "GEMM"
        or type == "MADD"
    ) and (apd != 0)


def insert_a_row(df, df_row, idx):
    df1 = df.iloc[:idx, :]
    df2 = df.iloc[idx:, :]
    df_new = pd.concat([df1, df_row, df2], ignore_index=True)
    return df_new
