using MultivariateStats;
using CSV;
using DataFrames;
using Random;
using ScikitLearn;
using ScikitLearn: fit!, predict;
using Plots;
@sk_import preprocessing: LabelEncoder;
@sk_import metrics: accuracy_score;
@sk_import model_selection: train_test_split;
@sk_import tree: DecisionTreeClassifier;
@sk_import svm: SVC;

#Load Dataframe
df = DataFrame(CSV.File("01_hdp.csv"));
attributes = convert(Array, df[:,:]);
#Shuffle rows for training/testing split
attributes = attributes[shuffle(1:end), :];
Y = attributes[:,15];

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

#Remove cancer stage row
allX = vcat(column_major[1:14, :], column_major[16:27, :]);

tree_accuracies = zeros(20);
svc_accuracies = zeros(20);
i = 1;
for entry in tree_accuracies
    println(i, " Dimension(s)");
    #Create and use the PCA model to reduce dimensions
    M = fit(PCA, allX; maxoutdim=i);
    smalldim_col = MultivariateStats.transform(M, allX);

    #Convert result to row major for classification algorithms
    smalldim_row = transpose(smalldim_col);

    #Split the data into 70% training and 30% testing (5968 is ~70% of 8525)
    #println("test1")
    Xtr = smalldim_row[1:5968, :];
    Xte = smalldim_row[5969:8525, :];
    #println("test2");
    Ytr = Y[1:5968];
    Yte = Y[5969:8525];
    #println("test3");
    
    #Classification Trees
    println("\tTrees")
    # Create and train
    tree_model = DecisionTreeClassifier();
    start = time_ns();
    fit!(tree_model, Xtr, Ytr);
    stop = time_ns();
    println("\t\tTraining Time: ", float(stop-start) \ 1000000000);
    # Test
    start = time_ns();
    tree_predictions = predict(tree_model, Xte);
    stop = time_ns();
    println("\t\tTesting Time: ", float(stop-start) \ 1000000000);
    tree_acc = accuracy_score(tree_predictions, Yte)
    global tree_accuracies[i] = tree_acc
    println("\t\tAccuracy: ", tree_acc);

    #SVC
    println("\tSVC");
    # Create and train
    svm_model = SVC();
    start = time_ns();
    fit!(svm_model, Xtr, Ytr);
    stop = time_ns();
    println("\t\tTraining Time:", float(stop-start) \ 1000000000);
    # Test
    start = time_ns();
    svm_predictions = predict(svm_model, Xtr);
    stop= time_ns();
    println("\t\tTesting Time: ", float(stop-start) \ 1000000000);
    svc_acc = accuracy_score(svm_predictions, Ytr)
    global svc_accuracies[i] = svc_acc;
    println("\t\tAccuracy: ", svc_acc);
    
    global i = i+1;
end

x = 1:20;

plot(x, tree_accuracies, title = "Tree Accuracies");
savefig("tree_accuracies.pdf");
plot(x, svc_accuracies, title = "SVC Accuracies");
savefig("svc_accuracies.pdf");