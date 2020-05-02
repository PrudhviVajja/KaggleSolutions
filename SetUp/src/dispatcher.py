from sklearn import ensemble

clf1 = ensemble.RandomForestClassifier(n_estimators=200, n_jobs = -1, verbose = 2)
clf2 = ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs = -1, verbose = 2)
MODELS = {
    "randomforest": clf1,
    "extratrees": clf2
}