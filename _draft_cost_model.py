# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:03:58 2022

@author: AShumaker
"""

#import base packages
import os
import pandas as pd
import numpy as np
# Plotting
import matplotlib.pyplot as plt
from scipy import ndimage
#machine learning package
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor


# %%
#define filepaths
base_dir = os.getcwd()
#well data
file_name_well_data = 'src\df_dataset.csv'
path_dataset = os.path.join(base_dir, file_name_well_data)

def col_filter_keep(df_in, list_keep):
    df_out = df_in.drop(columns=df_in.columns.difference(list_keep))
    return df_out

df_dataset = pd.read_csv(path_dataset)
model_cols_keep = ['Lat_length','Proppant_per_Ft','Fluid_per_Ft','Max_Gas_per_Ft']
df_model = col_filter_keep(df_dataset, model_cols_keep)
df_model.sort_index(axis=1, ascending=True, inplace=True)

# %%

def shuffle_and_split(df_in, shuffle_bool, test_size, y_col):
    if shuffle_bool == True:
        df_shuffled = shuffle(df_in, random_state=13) # 1337
    else:
        df_shuffled = df_in
    split_index = int((1-test_size)*len(df_shuffled))
    df_train_set = df_shuffled.iloc[0:split_index,:]
    df_test_set = df_shuffled.iloc[split_index:,:]
    # %train/test split
    X_cols = df_in.columns.difference([y_col]).to_list()
    X_train = df_train_set[X_cols]
    y_train = df_train_set[y_col]
    # test
    X_test = df_test_set[X_cols]
    y_test = df_test_set[y_col]
    return(df_train_set, df_test_set, X_train, y_train, X_test, y_test)
# %%

df_train_set, df_test_set, X_train, y_train, X_test, y_test = shuffle_and_split(df_model, False, .05, 'Max_Gas_per_Ft')
# %%
dict_params = {'n_estimators':25,'max_depth':5,'max_features':3,'min_samples_split':.1}
model_gbm = GradientBoostingRegressor(**dict_params,random_state=0)
model_gbm.fit(X_train,y_train)

# %% Cost Model
c_prop = .06 #dollars per pound
c_fluid = 7 #dollars per barrell
c_lat = 200 #dollars per foot
dict_cost_params = {'Fluid_per_bbl':c_fluid, 'Cost_per_lat_ft':c_lat,'Proppant_per_lb':c_prop}
# %%
def cost_model(df_in, y_hat, dict_cost_params):
    df = df_in.copy()
    df['Tot_Fluid']=df['Lat_length']*df['Fluid_per_Ft']
    df['Fluid_Cost']=df['Tot_Fluid']*dict_cost_params['Fluid_per_bbl']
    df['Drilling_Cost']=df['Lat_length']*dict_cost_params['Cost_per_lat_ft']+.5*1000*1000
    df['Tot_Proppant']=df['Lat_length']*df['Proppant_per_Ft']
    df['Proppant_Cost']=df['Tot_Proppant']*dict_cost_params['Proppant_per_lb']
    df['Total_Cost']=df['Fluid_Cost']+df['Drilling_Cost']+df['Proppant_Cost']
    df['Total_Max_Gas']=df['Lat_length']*y_hat
    df['Dollars_per_Max_Gas']=df['Total_Cost']/df['Total_Max_Gas']
    Z = df['Dollars_per_Max_Gas'].values
    return Z
# %%
#explanation: designed for a 3 feature model, nsteps is meshgrid dimension
#purpose: finds the "optimum 'X' feature values to maximize y"
#returns a series of the max features of the meshgrid prediction

def model_3D_meshgrid(trained_model, X, y, feature1, feature2, feature3, nsteps, bool_mode_cost):
    #ranges
    xmin=X_train[feature1].min()
    xmax=X_train[feature1].max()
    ymin=X_train[feature2].min()
    ymax=X_train[feature2].max()
    zmin=X_train[feature3].min()
    zmax=X_train[feature3].max()
    #create meshgrid based on feature ranges
    xx, yy, zz = np.meshgrid(np.linspace(xmin,xmax,nsteps), 
                             np.linspace(ymin, ymax, nsteps), 
                             np.linspace(zmin, zmax, nsteps))
    #pseudo data (meshgrid input to model)
    df_meshgrid = pd.DataFrame({feature1:xx.ravel(),feature2:yy.ravel(),feature3:zz.ravel()})
    
    y_hat = trained_model.predict(df_meshgrid)
    if bool_mode_cost == True:
        Z = cost_model(df_meshgrid, y_hat, dict_cost_params)
        index = np.argmin(Z)
        optimized_val = Z[index]
        optimized_attributes = df_meshgrid.iloc[index,:]
    else:
        index = np.argmax(y_hat)
        optimized_val = y_hat[index]
        optimized_attributes = df_meshgrid.iloc[index,:]
    
    return optimized_attributes, optimized_val
# %%
#ensures features are in the same order as the model input (X_train)
dict_features = {key:key_str for key, key_str in zip(
    ['feature1', 'feature2', 'feature3'],X_train.columns)}

dict_args = {'trained_model':model_gbm,'X':X_train,'y':y_train,'nsteps':50,'bool_mode_cost':False}
dict_args.update(dict_features)

optimized_attributes, optimized_val = [np.round(x,1) for x in model_3D_meshgrid(**dict_args)]

# %%
# this function calculates the meshgrid of predictions for the heatmap
#inputs are the features to be cross plotted
#output are the xx, yy, Z meshgrid values

def model_2D_meshgrid(trained_model, X, y, str_feature1, str_feature2, str_feature3, ser_vals3, nsteps, bool_mode_cost):
    # feature ranges
    xmin=X_train[str_feature1].min()
    xmax=X_train[str_feature1].max()
    ymin=X_train[str_feature2].min()
    ymax=X_train[str_feature2].max()
    zmin=X_train[str_feature3].min()
    zmax=X_train[str_feature3].max()
    # meshgrid of feature1 and feature2
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,nsteps), np.linspace(ymin, ymax, nsteps))
    #thrid dimension is all constant values
    array_feature3 = np.ones_like(xx.ravel())*ser_vals3[str_feature3]
    dict_arrays = {str_feature1:xx.ravel(),str_feature2:yy.ravel(), str_feature3:array_feature3}
    #order dictionary in same as Xtrain
    dict_meshgrid = {xcol:dict_arrays[xcol] for xcol in X.columns}
    df_meshgrid = pd.DataFrame(dict_meshgrid)
    
    #delta_grid
    grid_d = 3
    grid_x = np.array([grid_d, -grid_d, 0, 0, 0, 0])
    grid_y = np.roll(grid_x, 2)
    grid_z = np.roll(grid_y, 4)
    dx = (xmax-xmin)/(nsteps-1)
    dy = (ymax-ymin)/(nsteps-1)
    dz = (zmax-zmin)/(nsteps-1)
    list_vals=[[ser_vals3[str_feature1], ser_vals3[str_feature2], ser_vals3[str_feature3] ]]
    for x, y, z in zip(grid_x, grid_y, grid_z):
        list_vals.append([dx*x+ser_vals3[str_feature1], dy*y+ser_vals3[str_feature2], dz*z+ser_vals3[str_feature3]])
    
    df_grid = pd.DataFrame(list_vals, columns=[str_feature1, str_feature2, str_feature3])
    df_grid = df_grid[X.columns]
    
    # infer on meshgrid
    y_hat_meshgrid = trained_model.predict(df_meshgrid)
    #infer on stencil around a given point
    y_hat_grid = trained_model.predict(df_grid)
    if bool_mode_cost == True:
        Z = cost_model(df_meshgrid, y_hat_meshgrid, dict_cost_params).reshape(xx.shape)
        y_hat_grid = cost_model(df_grid, y_hat_grid, dict_cost_params)
    else:
        Z = y_hat_meshgrid.reshape(xx.shape)
        y_hat_grid = y_hat_grid
    
    dict_y_hat ={'mean':np.round(np.mean(y_hat_grid),1),'P10':np.round(np.quantile(y_hat_grid,.1),1),
                  'P90':np.round(np.quantile(y_hat_grid,.9),1)}
    
    return xx, yy, Z, dict_y_hat
# %%
#These dictionaries define the inputs for the "heatmaps"
#heatmap 1
dict_kwargs1 = {'trained_model':model_gbm,'X':X_train,'y':y_train,
               'str_feature1':'Proppant_per_Ft','str_feature2':'Fluid_per_Ft','str_feature3':'Lat_length',
               'ser_vals3':optimized_attributes,'nsteps':50,'bool_mode_cost':False}
#heatmap 2
dict_kwargs2 = {'trained_model':model_gbm,'X':X_train,'y':y_train,
               'str_feature1':'Lat_length','str_feature2':'Fluid_per_Ft','str_feature3':'Proppant_per_Ft',
               'ser_vals3':optimized_attributes,'nsteps':50,'bool_mode_cost':False}
#heatmap 3
dict_kwargs3 = {'trained_model':model_gbm,'X':X_train,'y':y_train,
               'str_feature1':'Lat_length','str_feature2':'Proppant_per_Ft','str_feature3':'Fluid_per_Ft',
               'ser_vals3':optimized_attributes,'nsteps':50,'bool_mode_cost':False}
# %% test block
xx, yy, Z, dict_y_hat = model_2D_meshgrid(**dict_kwargs1)
# %%
#function to generate plots
def plot_meshgrid(meshgrid_args, plotting_args):
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    #heat map grid points
    xx, yy, Z, dict_y_hat = model_2D_meshgrid(**meshgrid_args)
    # Apply gaussian filter for smoother heatmap
    Z_filt = ndimage.filters.gaussian_filter(Z, [1,1], mode='reflect')
    # plot marker at max value
    val_x = meshgrid_args['ser_vals3'][meshgrid_args['str_feature1']]
    val_y = meshgrid_args['ser_vals3'][meshgrid_args['str_feature2']]
    val_z = meshgrid_args['ser_vals3'][meshgrid_args['str_feature3']]
    #Plot heatmap
    shw = ax.imshow(Z_filt, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   **plotting_args)
    #plot marker
    mrker = ax.plot(val_x, val_y, '+', markeredgecolor='darkslategray', markeredgewidth=3, ms=16)
    #plot text
    txt = ax.annotate('    P90 {}\n    Mean {}\n    P10 {}'.format(dict_y_hat['P90'],dict_y_hat['mean'],dict_y_hat['P10']),
                      xy=(val_x, val_y), xycoords='data', xytext=(val_x,val_y), va='center', weight='bold', color='darkslategray')
            #xy=(x1, y1), xycoords='data',
            #xytext=(x2, y2), textcoords='offset points',
    #txt = ax.text(val_x, val_y, 'Value = {}'.format(y_hat_point),ha="center", va="top", weight='bold')
    #Plot annotations
    ax.set_xlabel(meshgrid_args['str_feature1'],size=12)
    ax.set_ylabel(meshgrid_args['str_feature2'],size=12)
    ax.set_title('{} = {}'.format(meshgrid_args['str_feature3'],val_z),
                pad=10, size=16, weight='bold')
    #colorbar
    bar = fig.colorbar(shw)
    bar.set_label('Max Gas Rate/FT Lateral') 
    plt.savefig(fname='{}_heatmap.png'.format(meshgrid_args['str_feature3']), format='png', dpi=150)
# %%
    #Heatmap 1 thrid dimension is from max_features
dict_plotting_args1 = {'cmap':plt.cm.magma,'vmin':0,'vmax':175,'aspect':'auto',
                     'origin':'lower'}

dict_plotting_args2 = {'cmap':plt.cm.magma_r,'vmin':3,'vmax':7,'aspect':'auto',
                     'origin':'lower'}

plot_meshgrid(dict_kwargs1, dict_plotting_args1)


# %%
def cost_model_grid(trained_model, X, y, str_feature1, str_feature2, str_feature3, ser_vals3, nsteps):
    # feature 1 and 2 ranges
    xmin=X_train[str_feature1].min()
    xmax=X_train[str_feature1].max()
    ymin=X_train[str_feature2].min()
    ymax=X_train[str_feature2].max()
    # meshgrid of feature1 and feature2
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,nsteps), np.linspace(ymin, ymax, nsteps))
    #thrid dimension is all constant values
    array_feature3 = np.ones_like(xx.ravel())*ser_vals3[str_feature3]
    dict_arrays = {str_feature1:xx.ravel(),str_feature2:yy.ravel(), str_feature3:array_feature3}
    #order dictionary in same as Xtrain
    dict_meshgrid = {xcol:dict_arrays[xcol] for xcol in X.columns}
    df_meshgrid = pd.DataFrame(dict_meshgrid)
    # predict
    y_hat = trained_model.predict(df_meshgrid)
    #Z = y_hat.reshape(xx.shape)
    df_cost = df_meshgrid.copy()
    df_cost['Tot_Fluid']=df_cost['Lat_length']*df_cost['Fluid_per_Ft']
    df_cost['Fluid_Cost']=df_cost['Tot_Fluid']*dict_cost_model['Fluid_per_bbl']
    df_cost['Drilling_Cost']=df_cost['Lat_length']*dict_cost_model['Cost_per_lat_ft']+.5*1000*1000
    df_cost['Tot_Proppant']=df_cost['Lat_length']*df_cost['Proppant_per_Ft']
    df_cost['Proppant_Cost']=df_cost['Tot_Proppant']*dict_cost_model['Proppant_per_lb']
    df_cost['Total_Cost']=df_cost['Fluid_Cost']+df_cost['Drilling_Cost']+df_cost['Proppant_Cost']
    df_cost['Total_Max_Gas']=df_cost['Lat_length']*y_hat
    df_cost['Dollars_per_Max_Gas']=df_cost['Total_Cost']/df_cost['Total_Max_Gas']
    
    Z =df_cost['Dollars_per_Max_Gas'].values.reshape(xx.shape)
    return xx, yy, Z
# %%
#function to generate plots
def plot_meshgrid_cost(meshgrid_args, plotting_args):
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    xx, yy, Z = cost_model_grid(**meshgrid_args)
    shw = ax.imshow(Z, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   **plotting_args)
    ax.set_xlabel(meshgrid_args['str_feature1'],size=12)
    ax.set_ylabel(meshgrid_args['str_feature2'],size=12)
    val3 = meshgrid_args['ser_vals3'][meshgrid_args['str_feature3']]
    ax.set_title('{} = {}'.format(meshgrid_args['str_feature3'],val3),
                pad=10, size=16, weight='bold')
    #colorbar
    bar = fig.colorbar(shw)
    bar.set_label('CompletionCost/Yr1_Gas_Production [$/MCF]') 
# %%
    #Heatmap 1 thrid dimension is from max_features
dict_plotting_args = {'cmap':plt.cm.magma_r,'vmin':3,'vmax':7,'aspect':'auto',
                     'origin':'lower'}

plot_meshgrid_cost(dict_kwargs1, dict_plotting_args)
# %%
# %% filter Z
sigma_y = 1.0
sigma_x = 1.0
# Apply gaussian filter
sigma = [sigma_y, sigma_x]
Z_filt = sp.ndimage.filters.gaussian_filter(Pmean, sigma, mode='reflect')