from skimage import measure
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pathlib import Path
from bfio import BioReader, BioWriter, LOG4J, JARS
import math
from multiprocessing import cpu_count
import csv
import logging
import filepattern
import skimage

logger = logging.getLogger("evaluating")
logger.setLevel(logging.INFO)
UNITS = {'m':  10**3,
         'cm': 10**1,
         'mm': 1,
         'µm': 10**-3,
         'nm': 10**-6,
         'pm': 10**-9}

header = ['Image_Name','Class', 'TP', 'FP', 'FN', 'over_segmented', 'under_segmented', 'IoU','sensitivity','precision','false negative rate',\
		'false discovery rate','F-Scores (weighted) ','F1-Score/dice index','Fowlkes-Mallows Index']

totalStats_header = ['Class', 'TP', 'FP', 'FN', 'over_segmented', 'under_segmented', 'IoU','sensitivity','precision','false negative rate',\
		'false discovery rate','F-Scores (weighted) ','F1-Score/dice index','Fowlkes-Mallows Index']


def get_image(im, tile_size, X, Y, x_max, y_max):
	"""Get tiled images based on tile size and set all right and lower border cells to 0.

	Args:
		img: input image
		tile_size: size of tile
		X: total image size in X
		Y: total image size in Y
		x_max: maximum value of x
		y_max: maximum value of y

	Returns:
		tiled image
	"""

	## get all cell values on the bottom border (second last row)
	b1 = np.unique(im[im.shape[0]-2, 0:tile_size])
	## get all cell values on the right border (second last column)
	b3 = np.unique(im[0:tile_size, im.shape[1]-2])
	if x_max < X and y_max < Y:
		val = np.concatenate([b1, b3])
		border_values = np.unique(val[val>0])
		for i in border_values:
			im = np.where(im == i, 0, im)
	elif x_max == X and y_max < Y:
		val = np.concatenate([b1])
		border_values = np.unique(val[val>0])
		for i in border_values:
			im = np.where(im == i, 0, im)
	elif x_max < X and y_max == Y:
		val = np.concatenate([b3])
		border_values = np.unique(val[val>0])
		for i in border_values:
			im = np.where(im == i, 0, im)      
	elif x_max == X and y_max == Y:
		im = im             
	return im


def metrics(tp, fp, fn):
	"""Calculate evaluation metrics.

	Args:
	    tp: true positive
	    fp: false positive
	    fn: false negative
	    tn: true neagtive

	Returns:
	    iou: intersection over union
	    tpr: true positive rate/sensitivity
	    precision: precision
	    fnr: false negative rate
	    fdr: false discovery rate
	    fscore: weighted f scores
	    f1_score: f1 score/dice index
	    fmi: Fowlkes-Mallows Index
	"""
	iou = tp/(tp+fp+fn)
	fval = 1
	tpr = tp/(tp+fn)
	precision = tp/(tp+fp)
	fnr = fn/(tp+fn)
	fdr = fp/(fp+tp)
	fscore = ((1+fval**2)*tp)/((1+fval**2)*tp + (fval**2)*fn + fp)
	f1_score = (2*tp)/(2*tp + fn + fp)
	fmi = tp / math.sqrt((tp + fp) * (tp + fn))
	return iou, tpr, precision, fnr,fdr,fscore,f1_score,fmi


def find_over_under(dict_result, data):
	"""Find number of over and under segmented cells.

	Args:
		dict_result: dictionary containing predicted labels for each ground truth cell.
		data: data to be saved to csv file.

	Returns:
		data: updated csv data with "over" or "under" label assigned to over and under segmented cells.
		over_segmented: number of over segmented cells.
		under_segmented: number of under segmented cells.
	"""
	over_segmented = 0; under_segmented = 0
	labels = {}
	for key in dict_result: 
		value = dict_result[key]
		if len(value) == 1:
			labels[key] = value[0]
		if len(value) > 1:        
			over_segmented+=1
			data[key].append("over")

	dict_new = {}
	for key, value in labels.items():
		dict_new.setdefault(value, set()).add(key)
	res = filter(lambda x: len(x)>1, dict_new.values())
	for i in list(res):
		for ind in i:
			data[ind].append("under")
			under_segmented+=1

	return data, over_segmented, under_segmented


def evaluation(GTDir, PredDir, inputClasses, outDir, individualData, individualSummary, \
	totalStats, totalSummary, spatial_resolution_xy, radiusFactor, filePattern):
	"""Main function to Read input files and save results based on user input.

	Args:
		GTDir: Ground truth images
		PredDir: Predicted images
		inputClasses: Number of Classes
		outDir: output directory
		individualData: Boolean to calculate individual image statistics.
		individualSummary: Boolean to calculate summary of individual images.
		totalStats: Boolean to calculate overall statistics across all images.
		totalSummary: Boolean to calculate summary across all images.
		spatial_resolution_xy: Spatial resolution of image (in mm).
		radiusFactor: Importance of radius/diameter to find centroid distance. Should be between (0,2].
		filePattern: Filename pattern to filter data.
	"""
	GTDir = Path(GTDir)
	PredDir = Path(PredDir)
	fp = filepattern.FilePattern(PredDir,filePattern)
	radiusFactor = radiusFactor if 0 < radiusFactor <= 2 else 1
	filename = str(Path(outDir)/"result.csv")
	f = open(filename, 'w')
	writer = csv.writer(f)
	writer.writerow(header)

	if totalStats:
		TP =[0] * (inputClasses+1); FP = [0] * (inputClasses+1); FN = [0] *(inputClasses+1)
		total_over_segmented = [0] * (inputClasses+1)
		total_under_segmented = [0] * (inputClasses+1)

	if individualSummary:
		header_individualSummary = ['Image','class','mean centroid distance for TP', 'mean IoU for TP']
		filename_individualSummary = str(Path(outDir)/'individual_image_summary.csv')
		f_individualSummary = open(filename_individualSummary, 'w')
		writer_individualSummary = csv.writer(f_individualSummary)
		writer_individualSummary.writerow(header_individualSummary)

	if totalSummary:
		total_iou = [0] * (inputClasses+1)
		total_tpr = [0] * (inputClasses+1) 
		total_precision = [0] * (inputClasses+1) 
		total_fnr = [0] * (inputClasses+1)
		total_fdr = [0] * (inputClasses+1)
		total_fscore = [0] * (inputClasses+1)
		total_f1_score = [0] * (inputClasses+1)
		total_fmi = [0] * (inputClasses+1)
		total_files = 0

	try:
		for fP in fp():
			for PATH in fP:
				file_name = PATH.get("file")
			tile_grid_size = 1
			tile_size = tile_grid_size * 2048
			with BioReader(file_name,backend='python',max_workers=cpu_count()) as br_pred:
				with BioReader(Path(GTDir/file_name.name),backend='python',max_workers=cpu_count()) as br_gt:
					# Loop through z-slices
					logger.info('Evaluating image {}'.format(file_name))
					total_files+=1

					if individualData:
						header1 = ['distance_centroids','class', 'IoU', 'Actual Label' ,'Predicted Labels', "TP or FN", "over/under"]
						filename1 = str(Path(outDir)/'cells_')+str((file_name.name)+'.csv')
						f1 = open(filename1, 'w')
						writer1 = csv.writer(f1)
						writer1.writerow(header1)

					for cl in range(1,inputClasses+1):
						mean_centroid = 0
						mean_iou = 0
						totalCells = 0
						fn = 0; tp = 0; fp = 0
						for z in range(br_gt.Z):
							# Loop across the length of the image
							for y in range(0,br_gt.Y,tile_size):
								y_max = min([br_gt.Y,y+tile_size])
								for x in range(0,br_gt.X,tile_size):
									x_max = min([br_gt.X,x+tile_size])
									im_gt = np.squeeze(br_gt[y:y_max,x:x_max,z:z+1,0,0])
									im_pred = np.squeeze(br_pred[y:y_max,x:x_max,z:z+1,0,0])

									im_gt = get_image(im_gt, tile_size, br_gt.X, br_gt.Y, x_max, y_max).astype(int)
									im_pred = get_image(im_pred, tile_size, br_pred.X, br_pred.Y, x_max, y_max).astype(int)
									
									## save class values in a separate class array for predicted and ground truth image.
									if inputClasses > 1:
    									### labels are classes
										classArray_pred = im_pred
										classArray_gt = im_gt
									else:
    									### all labels belong to the same class
										classArray_gt = (im_gt > 0).astype("uint8") * 1
										classArray_pred = (im_pred > 0).astype("uint8") * 1

									props = skimage.measure.regionprops(im_pred)
									numLabels_pred = np.unique(im_pred)
									centroids_pred = np.zeros((len(numLabels_pred),2)) 
									i = 1
									for prop in props:
										centroids_pred[i] = prop.centroid[::-1]
										i+=1

									list_matches = []
									props = skimage.measure.regionprops(im_gt)
									numLabels_gt = np.unique(im_gt)
									centroids_gt = np.zeros((len(numLabels_gt),2))
									diameters = np.zeros((len(numLabels_gt)))
									i = 1
									for prop in props:
										centroids_gt[i] = prop.centroid[::-1]
										diameters[i] = prop.minor_axis_length
										i+=1

									dict_result = {}
									data = [None]*(numLabels_gt.max()+1)

									# import time
									# start = time.time()

									#### Slow from here
									#### Calculate tp, fp, fn using nearest neighbor comparison
									nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(centroids_pred)
									for i in range(1,len(centroids_gt)):
										distance, index = nbrs.kneighbors(np.array([centroids_gt[i]]))
										index = index.flatten()
										componentMask_gt = (im_gt == numLabels_gt[i]).astype("uint8") * 1
										dict_result.setdefault(numLabels_gt[i], [])
										for idx in index:
											componentMask_pred_ = (im_pred == numLabels_pred[idx]).astype("uint8") * 1
											if np.where(componentMask_pred_ > 0)[0].size > 2 and idx != 0:
												if componentMask_gt[int(centroids_pred[idx][1]), int(centroids_pred[idx][0])] == 1 \
													or componentMask_pred_[int(centroids_gt[i][1]), int(centroids_gt[i][0])] == 1:
													dict_result[numLabels_gt[i]].append(numLabels_pred[idx])

									for i in range(1,len(centroids_gt)):
										distance, index = nbrs.kneighbors(np.array([centroids_gt[i]]))
										index = index.flatten()
										componentMask_gt = (im_gt == numLabels_gt[i]).astype("uint8") * 1
										match = index[0]
										dis = distance.flatten()[0]
										componentMask_pred = (im_pred == numLabels_pred[match]).astype("uint8") * 1
										mask_gt = np.where(componentMask_gt == 1, classArray_gt, 0)
										mask_pred = np.where(componentMask_pred == 1, classArray_pred, 0)

										intersection = np.logical_and(componentMask_pred, componentMask_gt)
										union = np.logical_or(componentMask_pred, componentMask_gt)
										iou_score_cell = np.sum(intersection) / np.sum(union)
										
										### divide diameter by 2 to get radius
										if np.unique(mask_gt)[1] == cl:
											if dis < (diameters[i]/2)*radiusFactor and match not in list_matches:
												if np.unique(mask_gt)[1] == np.unique(mask_pred)[1] and iou_score_cell > 0.5:
													tp+=1
													list_matches.append(match)
													condition = "TP"
													im_pred = np.where(im_pred==numLabels_pred[match], 0, im_pred)
													centroids_pred[match] = ([0.0,0.0])
													totalCells+=1
												else:
													fn+=1
													condition = "FN"
											else:
												fn+=1
												condition = "FN"

											if condition == "TP":
												mean_centroid+=dis
												mean_iou+=iou_score_cell

											data[numLabels_gt[i]] = [dis, cl, iou_score_cell,numLabels_gt[i], dict_result.get(numLabels_gt[i]), condition]
									### Slow till here

									data, over_segmented, under_segmented = find_over_under(dict_result, data)
									total_over_segmented[cl]+=over_segmented
									total_under_segmented[cl]+=under_segmented
									# end = time.time()
									# print(end-start)


									if individualData:
										for i in range(0, numLabels_gt.max()+1):
											if data[i] is not None:
												writer1.writerow(data[i])

									## calculate false positives.
									for i in range(1,len(centroids_pred)):
										if centroids_pred[i][0] != 0.0 and centroids_pred[i][1] != 0.0:
											componentMask_pred = (im_pred == numLabels_pred[i]).astype("uint8") * 1
											mask_pred = np.where(componentMask_pred == 1, classArray_pred, 0)
											if np.where(componentMask_pred > 0)[0].size > 2:
												if np.unique(mask_pred)[1] == cl:
													fp+=1

						# print(tp, fp, fn, over_segmented, under_segmented)
						iou, tpr, precision, fnr, fdr, fscore, f1_score, fmi = metrics(tp, fp, fn)
						data_result = [file_name.name, cl, tp, fp, fn, over_segmented, under_segmented,\
							 iou, tpr, precision, fnr,fdr,fscore,f1_score,fmi]
						writer.writerow(data_result)

						if totalSummary:
							total_iou[cl]+=iou
							total_tpr[cl]+=tpr 
							total_precision[cl]+=precision 
							total_fnr[cl]+=fnr
							total_fdr[cl]+=fdr
							total_fscore[cl]+=fscore
							total_f1_score[cl]+=f1_score
							total_fmi[cl]+=fmi

						if totalStats:
							TP[cl]+=tp
							FP[cl]+=fp
							FN[cl]+=fn

						if individualSummary:
							mean_centroid = mean_centroid/totalCells
							mean_iou = mean_iou/totalCells
							data_individualSummary = [file_name.name, cl, mean_centroid, mean_iou]
							writer_individualSummary.writerow(data_individualSummary)

					if individualData:
						f1.close()
		f_individualSummary.close()
		f.close()
		

		if totalSummary:
			totalSummary_header =  ['Class','Average IoU','Average sensitivity','Average precision','Average false negative rate',\
			'Average false discovery rate','Average F-Scores (weighted) ','Average F1-Score/dice index','Average Fowlkes-Mallows Index']
			summary_file = str(Path(outDir)/"total_summary.csv")
			f_totalSummary = open(summary_file, 'w')
			writer_totalSummary = csv.writer(f_totalSummary)
			writer_totalSummary.writerow(totalSummary_header)
			for cl in range(1,inputClasses+1):
				total_iou[cl] = total_iou[cl]/total_files
				total_tpr[cl] = total_tpr[cl]/total_files
				total_precision[cl] = total_precision[cl]/total_files 
				total_fnr[cl] = total_fnr[cl]/total_files
				total_fdr[cl] = total_fdr[cl]/total_files
				total_fscore[cl] = total_fscore[cl]/total_files
				total_f1_score[cl] = total_f1_score[cl]/total_files
				total_fmi[cl] = total_fmi[cl]/total_files
				data_totalSummary = [cl, total_iou[cl], total_tpr[cl], total_precision[cl],\
					 total_fnr[cl], total_fdr[cl], total_fscore[cl], total_f1_score[cl], total_fmi[cl]]
				writer_totalSummary.writerow(data_totalSummary)
			f_totalSummary.close()		

		if totalStats:
			overall_file = str(Path(outDir)/"total_stats_result.csv")
			f2 = open(overall_file, 'w')
			writer2 = csv.writer(f2)
			writer2.writerow(totalStats_header)
			for cl in range(1,inputClasses+1):
				iou, tpr, precision, fnr,fdr,fscore,f1_score,fmi = metrics(TP[cl], FP[cl], FN[cl])
				data_totalStats = [cl, TP[cl], FP[cl], FN[cl], total_over_segmented[cl], total_under_segmented[cl],\
					 iou, tpr, precision, fnr,fdr,fscore,f1_score,fmi]
				writer2.writerow(data_totalStats)
			f2.close()

	finally:
		logger.info('Evaluation complete.')
