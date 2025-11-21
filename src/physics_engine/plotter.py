import matplotlib.pyplot as plt


def plot_df(df, groupings):
    num_plots = len(groupings)

    ncols = 2
    nrows = 2

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
    for count, group in enumerate(groupings):
        col = count % ncols
        row = (count - count % ncols)//ncols
        #print(f"count: {count}, row: {row}, col: {col}")
        for component_name in group:
            axes[row, col].plot(df["t"], df[group][component_name])
        axes[row, col].legend(group)
    plt.tight_layout()
    plt.show()