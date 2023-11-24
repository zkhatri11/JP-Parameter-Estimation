import numpy,h5py
from matplotlib import pyplot, patches
from pycbc import conversions
from pycbc.results import scatter_histograms


file_name = 'GRMassPrior190521-single-M126-378-E-30-300.hdf'
with h5py.File(file_name, 'r') as f: ### M=[20,200], e3 = [-30,300]
    # Access datasets
    all_param_data = f['All_param'][:]
    M_valuesZ = f['Mass'][:]
    X_valuesZ = f['Spin'][:]
    e3_valuesZ = f['Epsilon'][:]

width_ratios = [2, 2, 2]
height_ratios = [2, 2, 2]

params = ['mass', 'spin','epsilon']
nparams = len(params)
lbls={'mass':' $M_f$',
      'spin':'$\chi_f$',
     'epsilon':'$\epsilon_3$'}
plot_colors = ['red','black','navy', 'lightseagreen', 'yellowgreen']

fig, axis_dict = scatter_histograms.create_axes_grid(
            params, labels=lbls,
            width_ratios=width_ratios, height_ratios=height_ratios,
            no_diagonals=False)
# {'mass':M_valuesD, 'spin':X_valuesD, 'epsilon': e3_valuesD},
all_samples = [{'mass':M_valuesZ, 'spin':X_valuesZ, 'epsilon': e3_valuesZ}]
legend_lbls = ['GW190521']

# Get the minimum and maximum of the mass for the plot limits
joint_masses = numpy.concatenate([s['mass'] for s in all_samples])
joint_epses = numpy.concatenate([s['epsilon'] for s in all_samples])
mins = {'mass':numpy.min(joint_masses), 'spin':0, 'epsilon':numpy.min(joint_epses)}
maxs = {'mass':numpy.max(joint_masses), 'spin':1, 'epsilon':numpy.max(joint_epses)}

handles = []
for si, samples in enumerate(all_samples):
    samples_color = plot_colors[si]

    # Plot 1D histograms
    for pi, param in enumerate(params):
        ax, _, _ = axis_dict[param, param]
        rotated = nparams == 2 and pi == nparams-1
        scatter_histograms.create_marginalized_hist(ax,
                samples[param], label=lbls[param],
                color=samples_color, fillcolor=None,
                linecolor=samples_color, title=True,
                rotated=rotated, plot_min=mins[param],
                plot_max=maxs[param], percentiles=[5,95])

    # Plot 2D contours
    ax, _, _ = axis_dict[('mass','spin')]
    scatter_histograms.create_density_plot(
                'mass', 'spin', samples, plot_density=False,
                plot_contours=True, percentiles=[90],
                contour_color=samples_color,
                xmin=mins['mass'], xmax=maxs['mass'],
                ymin=mins['spin'], ymax=maxs['spin'],
                ax=ax, use_kombine=False)
    ax, _, _ = axis_dict[('mass','epsilon')]
    scatter_histograms.create_density_plot(
                'mass','epsilon', samples, plot_density=False,
                plot_contours=True, percentiles=[90],
                contour_color=samples_color,
                xmin=mins['mass'], xmax=maxs['mass'],
                ymin=mins['epsilon'], ymax=maxs['epsilon'],
                ax=ax, use_kombine=False)
    ax, _, _ = axis_dict[('spin','epsilon')]
    scatter_histograms.create_density_plot(
                'spin','epsilon', samples, plot_density=False,
                plot_contours=True, percentiles=[90],
                contour_color=samples_color,
                xmin=mins['spin'], xmax=maxs['spin'],
                ymin=mins['epsilon'], ymax=maxs['epsilon'],
                ax=ax, use_kombine=False)

    handles.append(patches.Patch(color=samples_color, label=legend_lbls[si]))

fig.legend(loc=(0.8,0.8),
        handles=handles, labels=legend_lbls)
fig.set_dpi(250)
fig.savefig('GW190521 JP Posteriors.png')
