# a bar plot with errorbars
import matplotlib.pyplot as plt


def plot_percent_bars(
        percents, errors=None,
        colors=(
            '#cd1111', '#3f8618', '#2c7eca', '#c9e127', '#8b408e', '#a7aba4'
            ),
        left_right_padding = 0.5, width = 0.35, space_between_bars = 1.5,
        xticklabels=None, y_limit = (40, 70)
        ):
    # percents = (20, 25, 15, 5)
    # errors = (2, 3, 5, 2)
    # colors=['r','g','b', 'y']
    # left_right_padding = 0.5
    # width = 0.35
    # space_between_bars = 1.5
    # xticklabels=None
    # y_limit = (0, 100)

    if xticklabels is None:
        xticklabels = []
        for i in range(len(percents)):
            xticklabels.append('ROI_' + str(i))

    spacing_bars = []

    counter = left_right_padding
    for i in range(len(percents)):
        spacing_bars.append(counter * width)
        counter += space_between_bars

    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_axisbelow(True)
    ax.xaxis.set_tick_params(width=2, length=12)
    ax.yaxis.set_tick_params(width=2, length=12)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # opacity = 1.0
    # opacity = 0.4
    error_config = {'ecolor': '0.0', 'capsize': 10.0}

    bars = []
    for i in range(len(percents)):
        if errors is None:
            bars.append(
                ax.bar(
                    spacing_bars[i], percents[i], width, color=colors[i]
                    )
                )
        else:
            bars.append(
                ax.bar(
                    spacing_bars[i], percents[i], width,
                    color=colors[i], edgecolor=colors[i],
                    yerr=errors[i], error_kw=error_config,
                    )
                )

    plt.axhline(52.73, linewidth=4, color='k')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy using ROIs')
    ax.set_xticks([width * (i/width+left_right_padding) for i in spacing_bars])
    ax.set_xticklabels(xticklabels)

    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size':  16}
    plt.rc('font', **font)

    plt.tight_layout()

    # ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))

    plt.xlim(
        0.0,
        len(percents) * (width + left_right_padding*width) +
        left_right_padding*width
        )
    plt.ylim(y_limit)
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    plt.show()
