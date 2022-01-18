num_features = 6

def params_GenericSPEC():
    return {
        'k': list(range(1, num_features))
    }

def params_RobustScaler():
    return {
        'quantile_range':[(25.0, 75.0),(10.0, 90.0), (5.0, 95.0)],
        'with_centering': [True, False],
        'with_scaling': [True, False]
    }

def params_MyOutlierDetector():
    return {
        'n_neighbors':[32]
    }