using MultivariateStats;
using CSV;
using DataFrames;
using Random;
using ScikitLearn;
using ScikitLearn: fit!, predict;
@sk_import preprocessing: LabelEncoder;

#Load Dataframe
df = DataFrame(CSV.File("01_hdp.csv"));
attributes = convert(Array, df[:,:]);
#Shuffle rows for training/testing split
attributes = attributes[shuffle(1:end), :];

#Convert Family History
enc = LabelEncoder();
familyHx = enc.fit_transform(df[:,12]);
attributes[:,12] = familyHx;
#Convert Smoking
smoking = enc.fit_transform(df[:,13]);
attributes[:,13] = smoking/2;
#Convert Sex
sex = enc.fit_transform(df[:,14]);
attributes[:,14] = sex;
#Convert Cancer Stage
cancerStg = enc.fit_transform(df[:,15]);
attributes[:,15] = cancerStg/3;
#Convert School
school = enc.fit_transform(df[:,24]);
attributes[:,24] = school;

#Scale numerical Columns
attributes[:,1] = attributes[:,1] / 116.4579;
attributes[:,2] = attributes[:,2] / 2.128112;
attributes[:,3] = attributes[:,3] / 9;
attributes[:,4] = attributes[:,4] / 9;
attributes[:,5] = attributes[:,5] / 9;
attributes[:,6] = attributes[:,6] / 9;
attributes[:,7] = attributes[:,7] / 18;
attributes[:,10] = attributes[:,10] / 74.48235;
attributes[:,16] = attributes[:,16] / 10;
attributes[:,17] = attributes[:,17] / 9776.412;
attributes[:,18] = attributes[:,18] / 6.06487;
attributes[:,19] = attributes[:,19] / 58;
attributes[:,20] = attributes[:,20] / 23.72776;
attributes[:,21] = attributes[:,21] / 28.74211;
attributes[:,22] = attributes[:,22] / 407;
attributes[:,23] = attributes[:,23] / 29;
attributes[:,25] = attributes[:,25] / 9;
attributes[:,26] = attributes[:,26] / 35;
attributes[:,27] = attributes[:,27] / 0.81873;

floatAttr = convert(Array{Float64}, attributes);

#Find the transpose for Julia convention
column_major = transpose(floatAttr);

#Remove cancer stage row and make it the y-vector
allY = column_major[15,:];
allX = vcat(column_major[1:14, :], column_major[16:27, :])

#Create and use the PCA model to reduce dimensions
M = fit(PCA, allX; maxoutdim=10)
small_dim = MultivariateStats.transform(M, allX)

#Convert result to row major for classification algorithms
smalldim_row = transpose(smalldim_col);

#Split the data into 70% training and 30% testing (5968 is ~70% of 8525)
Xtr = smalldim_row[:, 1:5968];
Xte = smalldim_row[:, 5969:8525];
Ytr = smalldim_row[1:5968];
Yte = smalldim_row[5969:8525];
