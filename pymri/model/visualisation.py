import pylab as pl


def plot_haxby(activation, background, title, slice=25):
    z = slice

    pl.axis('off')
    pl.imshow(background[:, :, z].T, cmap=pl.cm.gray,
              interpolation='nearest', origin='lower')
    pl.imshow(activation[:, :, z].T, cmap=pl.cm.hot,
              interpolation='nearest', origin='lower')

    pl.title(title, x=.05, ha='left', y=.90, color='w', size=28)
    pl.show()
