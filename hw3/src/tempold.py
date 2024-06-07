def fill_mode(df, cols):
    for col_name in cols:
        mode_df = df.groupBy(col_name).count().orderBy(F.desc("count")).first()
        if mode_df and mode_df[0] is not None:
            mode = mode_df[0]
            df = df.na.fill({col_name: mode})
        else:
            # If mode is None, fill with a default value or skip
            default_value = 0  # Replace with a suitable default value if necessary
            df = df.na.fill({col_name: default_value})
    return df

def fill_median(df, cols):
    for col_name in cols:
        median = df.approxQuantile(col_name, [0.5], 0.01)[0]
        if median is not None:
            df = df.na.fill({col_name: median})
    return df

df = fill_mode(df, ['painloc', 'painexer', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope'])
df = fill_median(df, ['age', 'trestbps', 'thaldur', 'thalach', 'oldpeak'])