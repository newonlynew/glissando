import time
import logging
from pathlib import Path

import click

from .getter import FileType, MessagesGetter
from .embed import EmbeddingGenerator

import plotly.express as px
from umap import UMAP
import numpy as np
import hdbscan


@click.command()
@click.option(
    "-i",
    "--input-file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="File with the data",
)
@click.option(
    "-o",
    "--output-dir",
    show_default=True,
    default="./assets",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    help="Dir to save the plot",
)
@click.option(
    "-ft",
    "--filetype",
    show_default=True,
    default=FileType.STANDART.value,
    type=click.Choice([item.value for item in FileType]),
    help="Type of the data: standart/telegram/whatsapp",
)
def cli(input_file: Path, output_dir: Path, filetype: str) -> None:
    filetype = FileType(filetype)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
    )

    logger = logging.getLogger(__name__)

    start = time.time()
    getter = MessagesGetter.from_filetype(filetype)
    messages = getter.get_messages(input_file)
    logger.info(f"Messages loaded in {time.time() - start:.2f}s")

    start = time.time()
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(messages)
    logger.info(f"Embeddings generated in {time.time() - start:.2f}s")

    embeddings = embeddings.numpy()

    umap = UMAP(
        n_neighbors=20,  # Для 1000+ точек: 15-50
        min_dist=0.3,
        spread=2,
    )
    embeddings_2d = umap.fit_transform(embeddings)
    logger.info(f"UMAP projection completed in {time.time() - start:.2f}s")

    # Кластеризация HDBSCAN
    start = time.time()
    min_cluster_size = 10
    min_samples = 10
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    clusters = clusterer.fit_predict(embeddings_2d)
    logger.info(f"HDBSCAN clustering completed in {time.time() - start:.2f}s")
    logger.info(f"Found {len(np.unique(clusters)) - 1} clusters (plus noise)")

    # Визуализация с кластерами
    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color=clusters.astype(str),
        hover_name=[message.author + "<br>" + message.text for message in messages],
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )
    # Настройка отображения
    fig.update_traces(
        hovertemplate='%{hovertext}<extra></extra>',
        marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        legend_title_text='Cluster'
    )

    fig.show()

    output_dir.mkdir(exist_ok=True)
    fig.write_html(output_dir / "result.html")
    logger.info(f"Plot saved to {output_dir / 'result.html'}")