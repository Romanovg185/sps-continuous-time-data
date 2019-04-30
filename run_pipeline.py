from DataTransformationPipeline.convolve_simplified import find_participants_both, write_kernel_sum
from DataTransformationPipeline.four_box_plot import export_four_box_plots
from DataTransformationPipeline.generate_gml_graph import make_graphs
from DataTransformationPipeline.cell_indices_from_graphs import export_indices_correlating_cells

if __name__ == '__main__':
    find_participants_both()
    write_kernel_sum()
    export_four_box_plots()
    threshold_for_considering_correlation = 5
    make_graphs(threshold_for_considering_correlation)
    export_indices_correlating_cells()
