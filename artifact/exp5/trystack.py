import pandas as pd
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
# Your provided data
data_filckr = {
    'SpMM': [20.44, 5.35],
    'MatMul': [58.86, 74.90],
    'Format': [3.82, 0.00],
    'Others': [16.88, 19.75],
    'Tech': ['PyG', 'GS']
}


data_reddit2 = {
    'SpMM': [42.61, 70.54],
    'MatMul': [7.10, 19.44],
    'Format': [36.39, 0.00],
    'Others': [13.90, 10.02],
    'Tech': ['PyG', 'GS']
}

df1 = pd.DataFrame(data_filckr)
df1['Dataset'] = 'Flickr'
df2 = pd.DataFrame(data_reddit2)
df2['Dataset'] = 'Reddit2'

# merge the two dataframes
df = pd.concat([df1, df2], ignore_index=True)
# convert the dataframe to long format
df_long = pd.melt(df, id_vars=['Tech', 'Dataset'], var_name='Category', value_name='Percentage')
print(df_long)

# Now, plotting with seaborn.objects, facetting by 'Dataset'
plot = (
    so.Plot(df_long, x="Tech", y="Percentage", color="Category")
    .facet("Dataset")
    .add(so.Bar(), so.Stack()).scale(color="mako")
    .label(legend="Category", x="Technology", y="Percentage (%)")
)

plot.theme(axes_style("white"))
so.Plot.config.theme.update(axes_style("ticks"))

plot.save("breakdown.png")