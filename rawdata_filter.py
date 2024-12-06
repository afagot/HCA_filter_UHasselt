# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:54:11 2024

@author: afagot
"""

from pathlib import Path
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from permetrics import ClusteringMetric

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

# Access data location
atmDataPath =  'Data_filter/Data_atmcomp/'
treatedDataPath = 'Data_filter/Treated_data/'

# Data Biomasses
Biomasses = ['BSG', 'C', 'P']
BiomassesDict = {
    'BSG' : "Brewer'bio spent grain",
    'C' : "Carrot",
    'P' : "Potato"
}
nClusters = [[11,6,7,19,9,4,7],[9,7,4,13,12,9,16],[7,7,7,7,4,8,12]]
dropClusters = [
                 [[0,7],[2,4],[2,5],[0,4,8],[0],[0],[1,5]],
                 [[0,6],[0],[3],[8],[1],[1,7],[7,14]],
                 [[2],[2,4],[1],[3,4],[1],[1,5],[1,4]]
               ]

# For each data set...
for b, bio in enumerate(Biomasses):
    print('\nWorking with ' + BiomassesDict[bio] + '...\n')

    # # Loop 1 over the atm data files
    atmDataFiles = Path(atmDataPath).glob(bio+'*[1-3].csv')
    
    print('|>---------- LOOP OVER atm DATA FILES ----------<|')
    for f, file in enumerate(atmDataFiles):
    
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
        
        # Loop to find the best number of clusters for each sample for each 3
        # methods to compare
        max_clusters = 30  # max number of regions
        
        # Lists to store the data of the various scores
        X = []
        
        # Davies Bouldin
        avgDB = []
        
        # Dunn
        avgDunn = []
        
        for n in range(3, max_clusters+1):
            print("Number of clusters = ",n)
            X.append(n)
            # Average data
            avgWard = AgglomerativeClustering(compute_distances=True, n_clusters=n, linkage="ward")
            avgFit = np.reshape(data[9999].values,(-1,1))
            avgWard.fit(avgFit)
            #Compute Davies Bouldin Index
            avgDB.append(davies_bouldin_score(avgFit, avgWard.labels_))
            #Compute Dunn Index
            avgCM = ClusteringMetric(X=avgFit, y_pred=avgWard.labels_)
            avgDunn.append(avgCM.dunn_index())
        
        fig, ax = plt.subplots(nrows=2, ncols=1)
        
        # Plot the indexes for each clusterization of the sample
        ax[0].plot(X, avgDB)
        ax[0].set_ylabel("DB Index")
        ax[0].set_title(fName)
        
        ax[1].plot(X, avgDunn)
        ax[1].set_xlabel("Number of clusters")
        ax[1].set_ylabel("Dunn Index")
        plt.savefig(atmDataPath+fName+'-clustering-quality.pdf')
        plt.show()
        
        # Clusterize the data
        
        # Average data
        avgWard = AgglomerativeClustering(compute_distances=True, n_clusters=nClusters[b][f], linkage="ward")
        # avgWard = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="ward")
        avgFit = np.reshape(data[9999].values,(-1,1))
        avgWard.fit(avgFit)
        avgLabel = np.reshape(avgWard.labels_, avgHeatmap.shape)
        data.loc[:,99990] = avgWard.labels_
        
        # # plot the top five levels of the dendrogram
        # plot_dendrogram(avgWard, truncate_mode="level", p=6)
        # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        # plt.show()
        
        # Save the newcolumns to a new file
        data.to_csv(atmDataPath+fName+'-with-average.csv',index=False)
        
        # Print a heat map of the sample average
        figahm, axahm = plt.subplots()
        imahm = axahm.imshow(avgHeatmap, cmap='cividis', vmin=0, vmax=1.4, origin='lower')
        axahm.set_xlabel('x pixel')
        axahm.set_xticks([])
        axahm.set_ylabel('y pixel')
        axahm.set_yticks([])
        cbarahm = axahm.figure.colorbar(imahm, ax=axahm)
        cbarahm.ax.set_ylabel("Average absorbance (a.u.)", rotation=-90, va="bottom")
        plt.savefig(atmDataPath+fName+'-average-heatmap.pdf', bbox_inches='tight')
        plt.show()
        
        # Print a heat map of the sample average clustering
        figachm, axachm = plt.subplots()
        # get discrete colormap
        accmap = plt.get_cmap('cividis', np.max(avgLabel) - np.min(avgLabel) + 1)
        imachm = axachm.imshow(avgLabel, cmap=accmap, vmin=np.min(avgLabel) - 0.5, vmax=np.max(avgLabel) + 0.5, origin='lower')
        axachm.set_xlabel('x pixel')
        axachm.set_xticks([])
        axachm.set_ylabel('y pixel')
        axachm.set_yticks([])
        cbarachm = axachm.figure.colorbar(imachm, ticks=np.arange(np.min(avgLabel), np.max(avgLabel) + 1))
        cbarachm.ax.set_ylabel("Cluster membership", rotation=-90, va="bottom")
        plt.savefig(atmDataPath+fName+'-clustered-average-heatmap.pdf', bbox_inches='tight')
        plt.show()
        
        # Write data to new file and delete dataframe to free up memory
        del data
        
    # Loop 2 over the atm data files extended with the average column
    averageDataFiles = Path(atmDataPath).glob(bio+'*with-average.csv')

    print('|>------- LOOP OVER average DATA FILES --------<|')
    
    for f, file in enumerate(averageDataFiles):
    
        # Print file name to keep track while the code is running
        fName = str(os.path.basename(file)).split("-with-average.csv")[0]
        print(fName, file, sep=' - ')
        
        # Open the files
        data = pd.read_csv(file)
        data.columns = data.columns.astype(float)
        
        # Define the filters to apply on the average, peak and indicator values
        avgfilt = ~data[99990].isin(dropClusters[b][f])
        
        # Add a column Filter as a yes/no to create a map of where the data is.
        # If the filter is satisfied then it's 1, otherwise 0.
        data['Avg Filter'] = np.where(avgfilt, 1, 0)
        data['Avg Data'] = np.where(avgfilt, data[9999], 0)
        
        # Print a heat map of the filtered average sample
        figadhm, axadhm = plt.subplots()
        avgdataHeatmap = np.reshape(data['Avg Data'].values,(128,-1))
        imdhm = axadhm.imshow(avgdataHeatmap, cmap='cividis', vmin=0, vmax=1.4, origin='lower')
        axadhm.set_xlabel('x pixel')
        axadhm.set_xticks([])
        axadhm.set_ylabel('y pixel')
        axadhm.set_yticks([])
        cbaradhm = axadhm.figure.colorbar(imdhm, ax=axadhm)
        cbaradhm.ax.set_ylabel("Average absorbance (a.u.)", rotation=-90, va="bottom")
        plt.savefig(atmDataPath+fName+'-filtered-average-data-heatmap.pdf', bbox_inches='tight')
        plt.show()
        
        # Print a mask of the filtered average sample
        figamhm, axamhm = plt.subplots()
        avgMaskmap = np.reshape(data['Avg Filter'].values,(128,-1))
        imamhm = axamhm.imshow(avgMaskmap, cmap='cividis', vmin=0, vmax=1, origin='lower')
        axamhm.set_xlabel('x pixel')
        axamhm.set_xticks([])
        axamhm.set_ylabel('y pixel')
        axamhm.set_yticks([])
        plt.savefig(atmDataPath+fName+'-average-maskmap.pdf', bbox_inches='tight')
        plt.show()
        
        # Save this new column into a file in the treated data folder
        data.to_csv(treatedDataPath+fName+'-filter-flag.csv', index=False, columns=['Avg Filter'])
        
        del data
        
    #Loop 3 over the treated data to filter the noise and compute the average
    #spectrum of each sample. Filter them based on previous step.
    treatedDataFiles = Path(treatedDataPath).glob(bio+'*[1-3].csv')
    
    fig, ax = plt.subplots()
    
    print('|>-------- LOOP OVER TREATED DATA FILES --------<|')
    
    for i, file in enumerate(treatedDataFiles):
        
        # Print file name to keep track while the code is running
        fName = str(os.path.basename(file)).split(".")[0]
        print(fName, file, sep=' - ')
        
        # Get filter file name
        filterName = treatedDataPath + fName + '-filter-flag.csv'
        
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
        avg_meanData.to_csv(treatedDataPath+fName+'-average-mean.csv',header=False, index=True)
        
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
        
        del data
        del filterFlag
    
    plt.savefig(treatedDataPath+'Average-ESMC-data-spectra-'+bio+'.pdf', bbox_inches='tight')
    plt.show()