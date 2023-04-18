# PyMICE
PyMICE is a Python package that provides a new method for missing data imputation in datasets with multiple imputations using chained equations (MICE) </br>
It is more efficient and resource-saving than the package of sklearn =))

How to use that : 

```
def impute_with_mice(df, n_iterations=10):
    """
    Impute missing values using the MICE algorithm.

    Parameters:
    ----------
    df : pd.DataFrame
        The pandas dataframe containing the data to be imputed.
    n_iterations : int, optional
        The number of iterations to run the MICE algorithm. The default value is 10.

    Returns:
    -------
    imputed_df : pd.DataFrame
        The pandas dataframe with imputed values.
    """

    # Create a new instance of the MiceImputer class
    imputer = MiceImputer()

    # Impute missing values using the MICE algorithm and Bayesian Ridge regression
    imputed_df = pd.DataFrame(imputer.transform(df, BayesianRidge, n_iterations), columns=df.columns)

    return imputed_df
```

imputed_df = impute_with_mice(df, n_iterations=10)
