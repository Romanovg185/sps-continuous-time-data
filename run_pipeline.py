from DataTransformationPipeline.cast_to_csv import cast_to_csv
from DataTransformationPipeline.ground_truth import try_varying_ground_truths
from DataTransformationPipeline.convolve_simplified import find_participants_both, write_kernel_sum
from DataTransformationPipeline.four_box_plot import export_four_box_plots
from DataTransformationPipeline.generate_gml_graph import make_graphs
from DataTransformationPipeline.cell_indices_from_graphs import export_indices_correlating_cells

if __name__ == '__main__':
    recording_rate_hz = 30 # Hz
    ground_truth_sigma_range = (1, 5, 0.5) # Range of ground truth standard deviations to try out, in form (start, end, step)
    number_of_standard_deviations_threshold = 4 # Threshold on kernel sum to be applied
    threshold_for_considering_correlation = 5 # How many pairwise co-firing events have to be observed to be put into graph
    #cast_to_csv(recording_rate_hz)
    #print('1')
    try_varying_ground_truths(*ground_truth_sigma_range)
    #find_participants_both(number_of_standard_deviations_threshold)
    #print('2')
    #write_kernel_sum()
    #print('3')
    #export_four_box_plots()
    #print('4')
    #make_graphs(threshold_for_considering_correlation)
    #print('5')
    export_indices_correlating_cells()
    print('6')
