import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_vehicle_composition(csv_path, output_path=None):
    """Produce a pie chart of vehicle type composition using real data values."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    mapping = {
        'diesel_van': 'Diesel Van',
        'electric_van': 'Electric Van',
        'diesel_truck': 'Diesel Truck',
        'motorcycle': 'Motorcycle',
    }

    df['vehicle_label'] = df['vehicle_type'].map(mapping).fillna(df['vehicle_type'].str.replace('_', ' ').str.title())
    counts = df['vehicle_label'].value_counts().sort_values(ascending=False)

    total = len(df)
    labels = [f"{label} ({count}, {count/total:.1%})" for label, count in counts.items()]

    colors = [
        '#0f766e',  # Diesel Van
        '#2dd4bf',  # Electric Van
        '#0f172a',  # Diesel Truck
        '#66788a',  # Motorcycle
    ]

    # If additional categories exist, extend colors as needed
    if len(counts) > len(colors):
        extra = len(counts) - len(colors)
        colors = colors + list(plt.cm.tab20.colors[:extra])

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        colors=colors[:len(counts)],
        autopct='%1.1f%%',
        startangle=140,
        textprops=dict(color='black', fontsize=10),
        wedgeprops=dict(edgecolor='white', linewidth=1)
    )

    ax.set_title(f"Dataset Composition (n={total})", fontsize=16, pad=20)

    ax.axis('equal')

    plt.legend(counts.index, title='Vehicle Type', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=180, bbox_inches='tight')
        print(f"Saved pie chart to {output_path}")

    plt.show()


if __name__ == '__main__':
    csv_file = os.path.join(os.path.dirname(__file__), 'data', 'synthetic_deliveries.csv')
    out_file = os.path.join(os.path.dirname(__file__), 'vehicle_distribution_pie.png')

    plot_vehicle_composition(csv_file, out_file)
