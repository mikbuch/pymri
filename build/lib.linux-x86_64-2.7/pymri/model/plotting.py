import pylab as pl


def plot_haxby(activation, background, title):
    z = 25

    fig = pl.figure(figsize=(4, 5.4))
    fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
    pl.axis('off')
    pl.imshow(background[:, :, z].T, cmap=pl.cm.gray,
              interpolation='nearest', origin='lower')
    pl.imshow(activation[:, :, z].T, cmap=pl.cm.hot,
              interpolation='nearest', origin='lower')

    pl.title(title, x=.05, ha='left', y=.90, color='w', size=28)
