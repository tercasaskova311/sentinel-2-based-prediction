import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_pyplot_theme():
    plt.style.use('seaborn-v0_8-paper')
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.dpi": 300
    })

def plot_landcover_distributon(df: pd.DataFrame):
    """
    Plot the land cover distribution in Šumava National Park
    Args:
        df: pandas DataFrame with the land cover distribution
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['class_name'], df['area_km2'], color=sns.color_palette('Set2', len(df)))

    ax.set_yscale('log')
    ax.set_ylabel('Area (km²) - Log Scale', fontsize=12)
    ax.set_xlabel('Land Cover Class', fontsize=12)
    ax.set_title('Land Cover Distribution in Šumava National Park', pad=20, fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    # Show area + percentage on each bar
    for bar, area, pct in zip(bars, df['area_km2'], df['area_percentage']):
        if area > 0: 
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                area * 1.08,
                f'{area:.2f} km²\n({pct:.2f}%)',
                ha='center',
                va='bottom',
                fontsize=9
            )
    ax.margins(y=0.2)

    plt.tight_layout()
    plt.show()

def plot_disturbance_area_and_num_polygons_per_year(gdf, file_path):
    """
    Plot disturbance polygons total area and number of polygons per year
    Args:
        gdf: pandas DataFrame with the disturbance polygons
        file_path: str, path to save the figure
    Returns:
        None
    """
    gdf_per_year = gdf.groupby('year').agg(
        n_polygons=('id', 'count'),
        area_km2=('area_m2', lambda x: x.sum() / 1e6)
    )


    count_by_year = gdf_per_year['n_polygons']
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Disturbance Polygons in Sumava NP (2006-2024)', fontsize=16)
    count_by_year.plot(kind='line', marker='o', color='orange', ax=ax[0])
    ax[0].set_title('Number of Disturbance Polygons per Year')
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel('Number of Polygons')
    ax[0].set_xticks(ticks=count_by_year.index)
    ax[0].set_xticklabels(labels=count_by_year.index.astype(str), rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    area_by_year = gdf_per_year['area_km2']
    area_by_year.plot(kind='bar', color='green', ax=ax[1])
    ax[1].set_title('Total Disturbance Area per Year')
    ax[1].set_xlabel('Year', fontsize=12)
    ax[1].set_ylabel('Disturbance Area (km²)', fontsize=12)
    ax[1].set_xticklabels(labels=area_by_year.index.astype(str), rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()


def plot_area_and_count_by_cause(df):
    """
    Plot total deforested area and number of disturbance polygons by cause
    Args:        df: pandas DataFrame with the disturbance polygons and their causes
    Returns:        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle('Disturbance Area and Count by Cause', fontsize=16)
    disturbance_by_cause = df.groupby('label_disc')
    area_by_cause = disturbance_by_cause['area_m2'].sum() / 1e6
    area_by_cause.plot(kind='bar', color='coral', ax=ax[0])
    ax[0].set_title('Total Deforested Area by Cause')
    ax[0].set_xlabel('Cause of Deforestation')
    ax[0].set_ylabel('Deforested Area (km²)')
    ax[0].set_xticklabels(labels=area_by_cause.index, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    count_by_cause = disturbance_by_cause['id'].count()
    count_by_cause.plot(kind='bar', color='skyblue', ax=ax[1])
    ax[1].set_title('Number of Disturbance Polygons by Cause')
    ax[1].set_xlabel('Cause of Deforestation')
    ax[1].set_ylabel('Number of Polygons')
    ax[1].set_xticklabels(labels=count_by_cause.index, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()