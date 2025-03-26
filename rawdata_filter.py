# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:54:11 2024

@author: afagot
"""

from pathlib import Path
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
import ast
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from permetrics import ClusteringMetric

# Define the functions that we'll be using

#------------------------------------------------------------------------------
# plot_dendrogram: Create linkage matrix and then plot the dendrogram of the
# HCA. The clustering is obtained via sklearn.cluster's AgglomerativeClustering.
# With the arguments truncate_mode = "level" and p=nClusters you can
# decide how far you want to go down the tree.
#
# Example to plot the top five levels of the dendrogram of a clustering Clust:
#
# Clust = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="ward")
# plot_dendrogram(Clust, truncate_mode="level", p=6)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()
#------------------------------------------------------------------------------

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#------------------------------------------------------------------------------
# getOptimalNClusters: will increment the number of clusters from 3 to
# maxClusters and compute the Davies Bouldin index and Dunn index. It will
# create a plot to see the variation of both scores as well as a file with
# the stored values for later use if needed.
#------------------------------------------------------------------------------

def get_optimal_nClusters(pd_dataframe, max_clusters, data_dir, fileName):
    
    # Lists to store the data of the various scores
    N = [n for n in range(3, max_clusters+1)]   # number of clusters
    DB = [N, []]                                # Davies Bouldin
    Dunn = [N, []]                              # Dunn
    
    for n in range(3, max_clusters+1):
        print("Number of clusters = ",n)
        
        # Perform HCA on average data
        avgWard = AgglomerativeClustering(compute_distances=True, n_clusters=n, linkage="ward")
        avgFit = np.reshape(pd_dataframe[9999].values,(-1,1))
        avgWard.fit(avgFit)
        
        #Compute Davies Bouldin Index
        DB[1].append(davies_bouldin_score(avgFit, avgWard.labels_))
        
        #Compute Dunn Index
        Dunn[1].append(ClusteringMetric(X=avgFit, y_pred=avgWard.labels_).dunn_index())
    
    # Write the indexes to a csv file
    with open(data_dir+fileName+'-indexes-scores.csv', 'w') as output:
        output.write('\n'.join(str(N[i])+','+str(DB[1][i])+','+str(Dunn[1][i]) for i in range(0, max_clusters-2)))
    
    # Make and save the corresponding plot
    fig, ax = plt.subplots(nrows=2, ncols=1)
    
    # Plot the indexes for each clusterization of the sample
    ax[0].plot(DB[0], DB[1])
    ax[0].set_ylabel("DB Index")
    ax[0].set_title(fileName)
    
    ax[1].plot(Dunn[0], Dunn[1])
    ax[1].set_xlabel("Number of clusters")
    ax[1].set_ylabel("Dunn Index")
    plt.savefig(data_dir+fileName+'-clustering-quality.pdf')
    plt.show()

#------------------------------------------------------------------------------
# clusterize_data: takes a FTIR microspec file and clusterizes it. Plots 2 kinds
# of heatmaps : 1 of the average absorbance throughout the sample and 1 of the
# clustering. The function also adds 2 columns to the data files: the average
# absorbance of the pixel and the cluster it belongs too.
#------------------------------------------------------------------------------

def clusterize_data(f, file, data_dir, n_clusters, get_optimal=False, max_clusters=30):
    # Print file name to keep track while the code is running
    fName = str(os.path.basename(file)).split(".")[0]
    print(fName, file, sep=' - ')
    
    # Open the files
    data = pd.read_csv(file)
    data.columns = data.columns.astype(float)
    
    # Arrays to keep the average value
    Average = []
    
    # Loop over the different pixel spectra in the file
    for index, row in data.iterrows():
        y = row.to_numpy()
        
        # Calculate the average of the spectra (total spectra and peaks)
        Average.append(sum(y)/len(y))
    
    # Save the average data in the dataframe and then save it to the file.
    # Use the column code 9999. The code is out of bounds for the wavelength
    # to easily retrieve it for future filtering use.
    data.loc[:,9999] = pd.Series(Average, index=data.index)
    
    # Get and reshape the columns to clusterize and make heatmaps
    avgHeatmap = np.reshape(data[9999].values,(128,-1))
    
    # Find the best number of clusters for each sample
    if get_optimal:
        get_optimal_nClusters(data, max_clusters, data_dir, fName)
    
    # Clusterize the data
    
    # Average data
    avgWard = AgglomerativeClustering(compute_distances=True, n_clusters=n_clusters, linkage="ward")
    avgFit = np.reshape(data[9999].values,(-1,1))
    avgWard.fit(avgFit)
    avgLabel = np.reshape(avgWard.labels_, avgHeatmap.shape)
    data.loc[:,99990] = avgWard.labels_
    
    # Save the newcolumns to a new file
    data.to_csv(data_dir+fName+'-with-average.csv',index=False)
    
    # Print a heat map of the sample average
    figahm, axahm = plt.subplots()
    imahm = axahm.imshow(avgHeatmap, cmap='cividis', vmin=0, vmax=1.4, origin='lower', extent=(0.5, 128.5, 0.5, 128.5))
    axahm.set_xlabel('x pixel (2.76 '+r"$\mu m)$")
    axahm.set_xticks([1,32,64,96,128])
    axahm.set_ylabel('y pixel (2.77 '+r"$\mu m)$")
    axahm.set_yticks([1,32,64,96,128])
    cbarahm = axahm.figure.colorbar(imahm, ax=axahm)
    cbarahm.ax.set_ylabel("Average absorbance (a.u.)", rotation=-90, va="bottom")
    plt.savefig(data_dir+fName+'-average-heatmap.pdf', bbox_inches='tight')
    plt.show()
    
    # Print a heat map of the sample average clustering
    figachm, axachm = plt.subplots()
    # get discrete colormap
    accmap = plt.get_cmap('cividis', np.max(avgLabel) - np.min(avgLabel) + 1)
    imachm = axachm.imshow(avgLabel, cmap=accmap, vmin=np.min(avgLabel) - 0.5, vmax=np.max(avgLabel) + 0.5, origin='lower', extent=(0.5, 128.5, 0.5, 128.5))
    axachm.set_xlabel('x pixel (2.76 '+r"$\mu m)$")
    axachm.set_xticks([1,32,64,96,128])
    axachm.set_ylabel('y pixel (2.77 '+r"$\mu m)$")
    axachm.set_yticks([1,32,64,96,128])
    cbarachm = axachm.figure.colorbar(imachm, ticks=np.arange(np.min(avgLabel), np.max(avgLabel) + 1))
    cbarachm.ax.set_ylabel("Cluster membership", rotation=-90, va="bottom")
    plt.savefig(data_dir+fName+'-clustered-average-heatmap.pdf', bbox_inches='tight')
    plt.show()
    
    # Write data to new file and delete dataframe to free up memory
    del data

#------------------------------------------------------------------------------
# filter_data: uses the average absorbance and cluster assignement to filter the
# data and adds the filter flag and the corresponding filtered average absorbance
# to compute later the average spectrum of the sample. While running, the function
# also plots a few heat maps : the heatmap of the filtered average absorbance and
# the heatmap corresponding to the filter mask.
#------------------------------------------------------------------------------

def filter_data(f, file, data_path, output_path, drop_clusters=True, clusters_to_drop=[]):
    # Print file name to keep track while the code is running
    fName = str(os.path.basename(file)).split("-with-average.csv")[0]
    print(fName, file, sep=' - ')
    
    # Open the files
    data = pd.read_csv(file)
    data.columns = data.columns.astype(float)
    
    # Define the filters to apply on the average, peak and indicator values
    if(drop_clusters):
        avgfilt = ~data[99990].isin(clusters_to_drop)
    
    # Add a column Filter as a yes/no to create a map of where the data is.
    # If the filter is satisfied then it's 1, otherwise 0.
    data['Avg Filter'] = np.where(avgfilt, 1, 0)
    data['Avg Data'] = np.where(avgfilt, data[9999], 0)
    
    # Print a heat map of the filtered average sample
    figadhm, axadhm = plt.subplots()
    avgdataHeatmap = np.reshape(data['Avg Data'].values,(128,-1))
    imdhm = axadhm.imshow(avgdataHeatmap, cmap='cividis', vmin=0, vmax=1.4, origin='lower', extent=(0.5, 128.5, 0.5, 128.5))
    axadhm.set_xlabel('x pixel (2.76 '+r"$\mu m)$")
    axadhm.set_xticks([1,32,64,96,128])
    axadhm.set_ylabel('y pixel (2.77 '+r"$\mu m)$")
    axadhm.set_yticks([1,32,64,96,128])
    cbaradhm = axadhm.figure.colorbar(imdhm, ax=axadhm)
    cbaradhm.ax.set_ylabel("Average absorbance (a.u.)", rotation=-90, va="bottom")
    plt.savefig(data_path+fName+'-filtered-average-data-heatmap.pdf', bbox_inches='tight')
    plt.show()
    
    # Print a mask of the filtered average sample
    figamhm, axamhm = plt.subplots()
    avgMaskmap = np.reshape(data['Avg Filter'].values,(128,-1))
    axamhm.imshow(avgMaskmap, cmap='cividis', vmin=0, vmax=1, origin='lower', extent=(0.5, 128.5, 0.5, 128.5))
    axamhm.set_xlabel('x pixel (2.76 '+r"$\mu m)$")
    axamhm.set_xticks([1,32,64,96,128])
    axamhm.set_ylabel('y pixel (2.77 '+r"$\mu m)$")
    axamhm.set_yticks([1,32,64,96,128])
    plt.savefig(data_path+fName+'-average-maskmap.pdf', bbox_inches='tight')
    plt.show()
    
    # Save this new column into a file in the treated data folder
    data.to_csv(output_path+fName+'-filter-flag.csv', index=False, columns=['Avg Filter'])
    
    # Write data to new file and delete dataframe to free up memory
    del data

#------------------------------------------------------------------------------
# get_average_spectra: uses the filter mask previously provided by filter_data
# to remove from the data all the pixels corresponding to noise (pixels without
# tissue) and computes an average spectrum from the spectra of the remaining
# pixels. For each biomass, the average spectrum of each sample is returned with
# its corresponding 95% confidence level interval to build a figure with all the
# samples' spectra.
#------------------------------------------------------------------------------

def get_average_spectra(f, file, data_path, output_path, ax):
    # Print file name to keep track while the code is running
    fName = str(os.path.basename(file)).split(".")[0]
    print(fName, file, sep=' - ')
    
    # Get filter file name
    filterName = data_path + fName + '-filter-flag.csv'
    
    # Open the files
    data = pd.read_csv(file)
    data = data.drop(['map_x', 'map_y', 'Start Time', 'Sample Name', 'Filename', 'Label'], axis=1)
    data.columns = data.columns.astype(float)
    
    # Define the filter
    filterFlag = pd.read_csv(filterName)
    data[9999] = filterFlag['Avg Filter']
    avgfilt = data[9999] == 1
    
    # Filter the data and drop the filter flag column to calculate average spectra
    avgfilt_data = data[avgfilt]
    avgfilt_data = avgfilt_data.drop(avgfilt_data.columns[-1:], axis=1) # last column
    xAvg = avgfilt_data.columns.values
    
    # Calculate the mean/std deviation spectra and save them to a file
    avg_mean_ds = avgfilt_data.mean(numeric_only=True)
    avg_std_ds = avgfilt_data.std(numeric_only=True)
    avg_meanData = pd.concat([avg_mean_ds,avg_std_ds],axis=1)
    avg_meanData.to_csv(output_path+fName+'-average-mean.csv',header=False, index=True)
    
    # Plot the data spectra for each sample
    ax.plot(xAvg, avg_mean_ds, label=fName)
    ax.fill_between(xAvg, avg_mean_ds-1.96*avg_std_ds, avg_mean_ds+1.96*avg_std_ds, alpha=0.2)
    # ax.ticklabel_format(style='sci',axis='y',scilimits=(-2,-2),useMathText=True)
    ax.set_xlabel("Wavelength "+r"$(cm^{-1})$")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_xlim([850,4050])
    ax.set_xlim(ax.get_xlim()[::-1])
    # ax.set_ylim([-0.2,0.2])
    # ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Write data to new file and delete dataframe to free up memory
    del data
    del filterFlag

#------------------------------------------------------------------------------
# main: Main function of the analysis
#------------------------------------------------------------------------------

def main(get_optimal=False, max_clusters=30):
    # Get data and output location
    with open("data_path.txt", "r+") as data_path_file:
        data_path = data_path_file.read()
    with open("output_path.txt", "r+") as output_path_file:
        output_path = output_path_file.read()
    
    # Get the list of biomasses
    biomasses = pd.read_csv("biomasses.csv", header=0)
    biomasses_list = biomasses['Short Name'].to_list()
    biomasses_dict = dict(biomasses.values)
    del biomasses
    
    # Get the number of samples
    n_samples = len(list(Path(data_path).glob(biomasses_list[0]+'*[1-3].csv')))
    
    # Check if a file with the number of clusters to be used for each sample
    # exists. If not, get optimal number of clusters by default.
    by_default_get_optimal = False
    n_clusters = pd.DataFrame(columns=biomasses_list, index=[i for i in range(0, n_samples)])
    if os.path.isfile(data_path+"n_clusters.csv"):
        n_clusters = pd.read_csv(data_path+"n_clusters.csv", header=0)
    else:
        by_default_get_optimal = True
    
    # Check if a file with the clusters corresponding to noise to drop for each
    # sample exists. If not, do not drop clusters.
    drop_clusters = True
    if os.path.isfile(data_path+"clusters_to_drop.csv"):
        clusters_to_drop = pd.read_csv(data_path+"clusters_to_drop.csv", sep=';', header=0)
    else:
        drop_clusters = False
    
    # For each data set...
    for b, bio in enumerate(biomasses_list):
        print('\nWorking with ' + biomasses_dict[bio] + '...\n')
    
        # Loop 1 over the atm data files
        print('|>---------- LOOP OVER raw DATA FILES ----------<|')
        
        data_files = Path(data_path).glob(bio+'*[1-3].csv')
        for f, file in enumerate(data_files):
            if by_default_get_optimal:
                get_optimal=True
            clusterize_data(f, file, data_path, n_clusters[bio].iloc[f], get_optimal=get_optimal, max_clusters=max_clusters)
            
            
        # Loop 2 over the atm data files extended with the average column
        print('|>------- LOOP OVER average DATA FILES --------<|')
        
        average_data_files = Path(data_path).glob(bio+'*with-average.csv')
        for f, file in enumerate(average_data_files):
            filter_data(f, file, data_path, output_path,
                        drop_clusters=drop_clusters,
                        clusters_to_drop=ast.literal_eval(clusters_to_drop[bio].iloc[f]))
            
        # #Loop 3 over the treated data to filter the noise and compute the average
        # #spectrum of each sample. Filter them based on previous step.
        print('|>-------- LOOP OVER TREATED DATA FILES --------<|')
        
        fig, ax = plt.subplots()
        
        treatedDataFiles = Path(output_path).glob(bio+'*[1-3].csv')
        for f, file in enumerate(treatedDataFiles):
            get_average_spectra(f, file, output_path, output_path, ax)
        
        plt.savefig(output_path+'Average-ESMC-data-spectra-'+bio+'.pdf', bbox_inches='tight')
        plt.show()

#------------------------------------------------------------------------------
# User's cose
#------------------------------------------------------------------------------

main()